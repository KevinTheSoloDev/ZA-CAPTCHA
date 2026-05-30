import os
import json
import yaml
import numpy as np
import argparse
from PIL import Image
from datetime import datetime
from scipy.spatial.distance import cosine
import onnxruntime as ort
import warnings

# Suppress ONNX platform warning
warnings.filterwarnings("ignore")

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class FlowerMatcher:
    def __init__(self, config):
        self.config = config
        
        model_path = os.path.join(
            config["paths"]["models"], "model.onnx"
        )
        classes_path = os.path.join(
            config["paths"]["models"], "classes.json"
        )

        # Check files exist
        for path in [model_path, classes_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"\n❌ File not found: {path}\n"
                    "💡 Run: git pull to get latest model!"
                )

        # Load class names
        with open(classes_path) as f:
            self.classes = json.load(f)
        print(f"✅ Classes loaded: {len(self.classes)}")

        # Load ONNX model
        print("⏳ Loading ONNX model...")
        self.session = ort.InferenceSession(
            model_path,
            providers=[
                'NnapiExecutionProvider',
                'XnnpackExecutionProvider',
                'CPUExecutionProvider'
            ]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.image_size = config["model"]["image_size"]
        print("✅ Model loaded successfully!")

    def preprocess(self, image_path):
        """
        Prepare image for ONNX model
        """
        img = Image.open(image_path).convert("RGB")
        img = img.resize(
            (self.image_size, self.image_size),
            Image.LANCZOS
        )
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array.astype(np.float32)

    def extract_features(self, image_path):
        """
        Extract feature vector from image using ONNX
        Returns the final layer output (softmax probabilities)
        """
        img_array = self.preprocess(image_path)
        
        # Run inference
        outputs = self.session.run(
            None,
            {self.input_name: img_array}
        )
        
        # Get output features (probability vector)
        features = outputs[0][0]
        return features

    def similarity(self, features1, features2):
        """
        Calculate cosine similarity between
        two feature vectors
        Returns 0-100% similarity score
        """
        # Handle zero vectors
        if np.all(features1 == 0) or np.all(features2 == 0):
            return 0.0

        cos_sim = 1 - cosine(features1, features2)
        
        # Clamp between 0 and 100
        score = max(0.0, min(100.0, cos_sim * 100))
        return round(float(score), 2)

    def find_match(self, query_path, options):
        """
        Find best matching image from options

        query_path : path to debug*.jpg (the question)
        options    : list of option image paths
        """
        print("\n" + "=" * 55)
        print("🔍 FLOWER MATCHING")
        print("=" * 55)
        print(f"📸 Query: {query_path}\n")

        # Check query image exists
        if not os.path.exists(query_path):
            return {
                "error": f"Query image not found: {query_path}"
            }

        # Extract query features
        print("⏳ Processing query image...")
        query_features = self.extract_features(query_path)
        
        # Show what the model thinks query is
        top_idx = np.argmax(query_features)
        print(
            f"🌸 Query identified as: "
            f"{self.classes[top_idx]} "
            f"({query_features[top_idx]*100:.1f}%)\n"
        )

        # Compare with each option
        print("⏳ Comparing with options...")
        print("-" * 55)
        
        results = []
        for i, option_path in enumerate(options, 1):
            if not os.path.exists(option_path):
                print(f"  ⚠️  Option {i} not found: {option_path}")
                continue

            option_features = self.extract_features(option_path)
            score = self.similarity(query_features, option_features)

            results.append({
                "option": i,
                "path": option_path,
                "filename": os.path.basename(option_path),
                "similarity": score
            })

            # Visual progress bar
            bar = "█" * int(score / 5)
            print(
                f"  Option {i}: "
                f"{os.path.basename(option_path):<25} "
                f"{score:>6.2f}% {bar}"
            )

        if not results:
            return {"error": "No valid option images found!"}

        # Find best match
        best = max(results, key=lambda x: x["similarity"])

        print("\n" + "=" * 55)
        print(f"✅ ANSWER: Option {best['option']}")
        print(f"   File  : {best['filename']}")
        print(f"   Score : {best['similarity']}%")
        print("=" * 55)

        return {
            "query": query_path,
            "timestamp": datetime.now().isoformat(),
            "version": self.config["project"]["version"],
            "results": results,
            "answer": best
        }


def print_error(message):
    print("\n" + "=" * 55)
    print(f"❌ ERROR: {message}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🌸 Flower Icon Matcher"
    )
    parser.add_argument(
        "query",
        help="Query image path (debug*.jpg)"
    )
    parser.add_argument(
        "options",
        nargs="+",
        help="Option image paths to compare against"
    )
    parser.add_argument(
        "--top", type=int, default=3,
        help="Show top N results (default: 3)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save results as JSON file"
    )
    args = parser.parse_args()

    # Validate minimum options
    if len(args.options) < 2:
        print_error("Please provide at least 2 option images!")
        exit(1)

    # Load config
    config = load_config()

    # Run matcher
    try:
        matcher = FlowerMatcher(config)
        results = matcher.find_match(
            query_path=args.query,
            options=args.options
        )

        # Handle errors
        if "error" in results:
            print_error(results["error"])
            exit(1)

        # Save results if requested
        if args.save:
            os.makedirs("results", exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output = f"results/match_{timestamp}.json"
            with open(output, "w") as f:
                json.dump(results, f, indent=4)
            print(f"\n💾 Results saved to {output}")

    except FileNotFoundError as e:
        print_error(str(e))
        exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        exit(1)
