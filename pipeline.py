#!/usr/bin/env python3
"""
Zerads Full Pipeline Solver
Downloads images → ONNX model → Extract CID → Submit
File-based integration with existing AI model
"""

import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import subprocess
import re
import time
import sys
import json
import logging
import socket
import os
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ZERADS_USER      = "Sologotemm"
TARGET_URL       = "https://antautosurf.com/"
FAIL_URL         = "https://antautosurf.com/"
ADS_NUM          = 10
STRICT           = 50

BASE_URL         = "https://zerads.com"
CAPTCHA_IMG_DIR  = "images/CaptchaPTC"
LOG_FILE         = "pipeline.log"
WAIT_TIME        = 8
MAX_CAPTCHAS     = 10
RETRY_LIMIT      = 3
BETWEEN_RUN_WAIT = 3
LINKS_TO_KEEP    = 10

# ── AI Model paths ──
# Edit these to match your setup
MODEL_SCRIPT     = Path("./matcher.py")        # your model script
CONFIG_PATH      = Path("./config/config.yaml")
IMAGE_DIR        = Path("./captcha_images")    # where images are saved

TOR_HOST         = "127.0.0.1"
TOR_SOCKS_PORT   = 9050
TOR_CONTROL_PORT = 9051
TOR_PASSWORD     = ""

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language":           "en-US,en;q=0.5",
    "Accept-Encoding":           "gzip, deflate, br",
    "Connection":                "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# AI MODEL INTERFACE
# ─────────────────────────────────────────────

class AIModel:
    """
    Interfaces with your ONNX flower matcher.

    Flow:
      debug1.jpg   → query image  (captcha.php)
      option1.jpg  → choice 1
      option2.jpg  → choice 2
      ...
      optionN.jpg  → choice N

    Model outputs: flower1.jpg / flower2.jpg etc
    We map filename number → CID from choices list
    """

    def __init__(self, script=MODEL_SCRIPT, image_dir=IMAGE_DIR):
        self.script    = Path(script)
        self.image_dir = Path(image_dir)

        if not self.script.exists():
            raise FileNotFoundError(
                f"Model script not found: {self.script}\n"
                f"Please set MODEL_SCRIPT path correctly"
            )

    def prepare_image_dir(self):
        """Clean old images before new captcha"""
        if self.image_dir.exists():
            deleted = 0
            for f in self.image_dir.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    f.unlink()
                    deleted += 1
            if deleted:
                log.info(f"  Cleaned {deleted} old image(s)")
        else:
            self.image_dir.mkdir(parents=True)

    def save_image(self, img_pil, filename):
        """Save PIL image to image_dir"""
        path = self.image_dir / filename
        img_pil.save(str(path), "JPEG", quality=95)
        log.info(f"  Saved: {filename} ({img_pil.size})")
        return path

    def run_model(self, query_path, option_paths):
        """
        Run your matcher.py script via subprocess.
        Returns the answer filename e.g. 'flower3.jpg'
        """
        cmd = [
            sys.executable,          # python
            str(self.script),         # matcher.py
            str(query_path),          # debug1.jpg
            *[str(p) for p in option_paths],  # option1..N.jpg
        ]

        log.info(f"  Running model...")
        log.debug(f"  CMD: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.script.parent)
            )

            stdout = result.stdout
            stderr = result.stderr

            if result.returncode != 0:
                log.error(f"  Model error: {stderr}")
                return None

            log.debug(f"  Model output:\n{stdout}")

            # Parse answer from output
            answer = self._parse_answer(stdout)
            return answer

        except subprocess.TimeoutExpired:
            log.error("  Model timed out after 60s")
            return None
        except Exception as e:
            log.error(f"  Model run failed: {e}")
            return None

    def _parse_answer(self, stdout):
        """
        Parse model output to get answer filename.

        Looks for lines like:
        '✅ ANSWER: Option 3'
        'File  : flower3.jpg'
        'Score : 94.5%'
        """
        answer = None

        # Method 1: Look for "ANSWER: Option N"
        m = re.search(r"ANSWER:\s*Option\s*(\d+)", stdout, re.I)
        if m:
            option_num = int(m.group(1))
            answer     = option_num
            log.info(f"  Model answer: Option {option_num}")

        # Method 2: Look for "File  : flowerN.jpg"
        if answer is None:
            m = re.search(
                r"File\s*:\s*flower(\d+)\.jpg", stdout, re.I
            )
            if m:
                answer = int(m.group(1))
                log.info(f"  Model file match: flower{answer}.jpg")

        # Method 3: Look for "flowerN.jpg" anywhere
        if answer is None:
            m = re.search(r"flower(\d+)\.jpg", stdout, re.I)
            if m:
                answer = int(m.group(1))
                log.info(f"  Model output match: flower{answer}.jpg")

        if answer is None:
            log.error(f"  Could not parse answer from:\n{stdout}")

        return answer

    def solve_captcha(self, captcha_img, option_imgs, choices):
        """
        Full solve pipeline:
        1. Clean image dir
        2. Save debug1.jpg + option1..N.jpg
        3. Run model
        4. Map option number → CID

        captcha_img : PIL Image (captcha.php)
        option_imgs : list of PIL Images (choices)
        choices     : list of CID integers matching option order
        """
        # Clean old images
        self.prepare_image_dir()

        # Save captcha target
        query_path = self.save_image(captcha_img, "debug1.jpg")

        # Save option images
        option_paths = []
        for i, img in enumerate(option_imgs, 1):
            path = self.save_image(img, f"option{i}.jpg")
            option_paths.append(path)

        log.info(
            f"  Images saved: debug1.jpg + "
            f"{len(option_paths)} options"
        )

        # Run model
        option_num = self.run_model(query_path, option_paths)

        if option_num is None:
            log.error("  Model returned no answer")
            return None

        # Map option number → CID
        # option1 → choices[0]
        # option2 → choices[1]
        # etc.
        idx = option_num - 1  # convert 1-based to 0-based

        if idx < 0 or idx >= len(choices):
            log.error(
                f"  Option {option_num} out of range "
                f"(have {len(choices)} choices)"
            )
            return None

        cid = choices[idx]
        log.info(
            f"  Option {option_num} → "
            f"choices[{idx}] → CID={cid}"
        )
        return cid


# ─────────────────────────────────────────────
# TOR CONTROLLER
# ─────────────────────────────────────────────

class TorController:

    def __init__(self):
        self.rotate_count = 0
        self.ip_history   = []

    def new_circuit(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((TOR_HOST, TOR_CONTROL_PORT))
            s.settimeout(10)
            if TOR_PASSWORD:
                s.send(f'AUTHENTICATE "{TOR_PASSWORD}"\r\n'.encode())
            else:
                s.send(b'AUTHENTICATE ""\r\n')
            s.recv(1024)
            s.send(b"SIGNAL NEWNYM\r\n")
            resp = s.recv(1024).decode()
            s.close()
            if "250" in resp:
                self.rotate_count += 1
                time.sleep(2)
                return True
            return False
        except Exception as e:
            log.error(f"Tor error: {e}")
            return False

    def get_ip(self):
        try:
            r = requests.get(
                "https://api.ipify.org?format=json",
                proxies={
                    "http":  f"socks5h://{TOR_HOST}:{TOR_SOCKS_PORT}",
                    "https": f"socks5h://{TOR_HOST}:{TOR_SOCKS_PORT}",
                },
                timeout=15
            )
            return r.json().get("ip", "?")
        except:
            return "?"

    def rotate(self):
        old = self.get_ip()
        log.info(f"Current IP: {old}")
        for attempt in range(8):
            self.new_circuit()
            time.sleep(2)
            new = self.get_ip()
            if new != old and new not in self.ip_history[-20:]:
                self.ip_history.append(new)
                log.info(f"🔄 {old} → {new}")
                return new
            old = new
            time.sleep(2)
        current = self.get_ip()
        self.ip_history.append(current)
        return current


# ─────────────────────────────────────────────
# SESSION MANAGER
# ─────────────────────────────────────────────

class SessionManager:

    def __init__(self, use_tor=True):
        self.use_tor  = use_tor
        self.last_url = None
        self.session  = self._make_session()

    def _make_session(self):
        s = requests.Session()
        s.headers.update(HEADERS)
        if self.use_tor:
            s.proxies.update({
                "http":  f"socks5h://{TOR_HOST}:{TOR_SOCKS_PORT}",
                "https": f"socks5h://{TOR_HOST}:{TOR_SOCKS_PORT}",
            })
        return s

    def get(self, url, extra_headers=None, **kwargs):
        headers = {}
        if self.last_url:
            headers["Referer"] = self.last_url
        if extra_headers:
            headers.update(extra_headers)
        try:
            r = self.session.get(
                url, headers=headers,
                timeout=20, allow_redirects=True,
                **kwargs
            )
            self.last_url = r.url
            return r
        except Exception as e:
            log.error(f"GET failed [{url}]: {e}")
            return None

    def get_image(self, url):
        """Fetch image as PIL using session cookies"""
        headers = {
            "Accept":         "image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Sec-Fetch-Dest": "image",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "same-origin",
        }
        if self.last_url:
            headers["Referer"] = self.last_url
        try:
            r = self.session.get(url, headers=headers, timeout=20)
            log.info(
                f"  Image: {r.status_code} | "
                f"{r.headers.get('Content-Type','?')} | "
                f"{len(r.content)}b"
            )
            if r.content.lstrip().startswith(b"<"):
                log.error("Got HTML instead of image")
                return None
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return img
        except Exception as e:
            log.error(f"Image fetch failed: {e}")
            return None


# ─────────────────────────────────────────────
# PAGE ANALYSER
# ─────────────────────────────────────────────

class PageAnalyser:

    def analyse(self, response):
        if response is None:
            return {
                "state": "error", "clicks_left": None,
                "captcha_data": None, "raw_url": ""
            }

        url  = response.url
        body = response.text
        soup = BeautifulSoup(body, "html.parser")

        result = {
            "state":        "unknown",
            "clicks_left":  None,
            "captcha_data": None,
            "raw_url":      url,
        }

        # Destination check
        if "antautosurf.com" in url and "zerads.com" not in url:
            result["state"] = "destination"
            return result

        meta = soup.find(
            "meta",
            attrs={"http-equiv": re.compile(r"refresh", re.I)}
        )
        if meta and "antautosurf.com" in meta.get("content", ""):
            result["state"] = "destination"
            return result

        for script in soup.find_all("script"):
            txt = script.get_text()
            if "antautosurf.com" in txt and (
                "window.location" in txt
                or "location.href" in txt
            ):
                result["state"] = "destination"
                return result

        # Captcha check
        captcha_data = self._extract_captcha(soup, url)
        if captcha_data:
            result["state"]        = "captcha"
            result["captcha_data"] = captcha_data
            result["clicks_left"]  = self._extract_clicks(body)
            return result

        # Error check
        for pat in [
            r"wrong\s+(captcha|answer|image)",
            r"incorrect\s+(captcha|answer)",
            r"invalid\s+captcha",
        ]:
            if re.search(pat, body, re.I):
                result["state"] = "error"
                return result

        return result

    def _extract_captcha(self, soup, page_url):
        tag = soup.find("img", src=re.compile(r"captcha\.php"))
        if not tag:
            return None

        src         = tag.get("src", "")
        captcha_url = (
            src if src.startswith("http")
            else f"{BASE_URL}/{src}"
        )

        params = {}
        if "?" in src:
            for part in src.split("?")[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v

        choices        = []
        shortlink_urls = {}
        option_img_urls = []   # URLs of the option images

        for a in soup.find_all(
            "a", href=re.compile(r"shortlink\.php")
        ):
            href = a.get("href", "")
            m    = re.search(r"cid=(\d+)", href)
            if not m:
                continue

            cid = int(m.group(1))
            if cid not in choices:
                choices.append(cid)
                shortlink_urls[cid] = (
                    href if href.startswith("http")
                    else f"{BASE_URL}/{href}"
                )

                # Get the img src inside this link
                img_tag = a.find("img")
                if img_tag:
                    img_src = img_tag.get("src", "")
                    img_url = (
                        img_src if img_src.startswith("http")
                        else f"{BASE_URL}/{img_src}"
                    )
                    option_img_urls.append(img_url)
                else:
                    option_img_urls.append(None)

        if not choices:
            return None

        return {
            "captcha_url":     captcha_url,
            "option_img_urls": option_img_urls,
            "choices":         choices,
            "shortlink_urls":  shortlink_urls,
            "params":          params,
            "page_url":        page_url,
        }

    def _extract_clicks(self, body):
        m = re.search(r"(\d+)\s+clicks?\s+remaining", body, re.I)
        return int(m.group(1)) if m else None


# ─────────────────────────────────────────────
# CHAIN SOLVER
# ─────────────────────────────────────────────

class ChainSolver:
    """
    Solves full captcha chain using AI model.
    Downloads captcha + option images,
    runs ONNX model, submits answer.
    """

    def __init__(self, session, analyser, ai_model):
        self.session   = session
        self.analyser  = analyser
        self.ai        = ai_model

    def solve(self, start_url):
        chain   = []
        attempt = 0

        response = self.session.get(start_url)
        if response is None:
            return self._result(False, chain, "Initial fetch failed")

        Path("debug_page_0.html").write_text(
            response.text, encoding="utf-8"
        )

        while attempt < MAX_CAPTCHAS:
            analysis = self.analyser.analyse(response)
            state    = analysis["state"]

            log.info(f"\n{'─'*50}")
            log.info(
                f"Step {attempt} | {state.upper()} | "
                f"clicks={analysis['clicks_left']} | "
                f"{response.url}"
            )
            log.info(f"{'─'*50}")

            if state == "destination":
                bypassed = len(chain) == 0
                if bypassed:
                    log.warning("⚠️  Bypassed - no captchas solved")
                else:
                    log.info(f"✅ Done in {len(chain)} captcha(s)")
                return self._result(True, chain, "", bypassed)

            if state in ("error", "unknown"):
                if "antautosurf.com" in response.url:
                    return self._result(
                        True, chain, "", len(chain) == 0
                    )
                Path(f"debug_fail_{attempt}.html").write_text(
                    response.text, encoding="utf-8"
                )
                return self._result(False, chain, f"State={state}")

            if state == "captcha":
                data    = analysis["captcha_data"]
                attempt += 1

                choices         = data["choices"]
                option_img_urls = data["option_img_urls"]

                log.info(f"\nCAPTCHA #{attempt}")
                log.info(f"  Choices         : {choices}")
                log.info(f"  Captcha URL     : {data['captcha_url']}")

                # Wait for JS timer
                log.info(f"  Waiting {WAIT_TIME}s...")
                time.sleep(WAIT_TIME)

                # ── Download captcha target image ──
                captcha_img = None
                for retry in range(1, RETRY_LIMIT + 1):
                    captcha_img = self.session.get_image(
                        data["captcha_url"]
                    )
                    if captcha_img:
                        break
                    log.warning(f"  Retry {retry}/{RETRY_LIMIT}...")
                    time.sleep(2)

                if captcha_img is None:
                    return self._result(
                        False, chain,
                        f"Captcha image fetch failed #{attempt}"
                    )

                # ── Download option images ──
                option_imgs = []
                for i, opt_url in enumerate(option_img_urls, 1):
                    if opt_url is None:
                        log.warning(f"  Option {i} URL missing")
                        option_imgs.append(None)
                        continue

                    opt_img = self.session.get_image(opt_url)
                    if opt_img:
                        option_imgs.append(opt_img)
                        log.info(f"  Option {i} downloaded")
                    else:
                        log.warning(f"  Option {i} download failed")
                        option_imgs.append(None)

                # Filter out None options and corresponding choices
                valid_options = [
                    (img, cid)
                    for img, cid in zip(option_imgs, choices)
                    if img is not None
                ]

                if not valid_options:
                    return self._result(
                        False, chain,
                        f"All option downloads failed #{attempt}"
                    )

                valid_imgs   = [v[0] for v in valid_options]
                valid_choices = [v[1] for v in valid_options]

                # ── Run AI model ──
                log.info(f"  Running AI model...")
                best_cid = self.ai.solve_captcha(
                    captcha_img,
                    valid_imgs,
                    valid_choices
                )

                if best_cid is None:
                    return self._result(
                        False, chain,
                        f"AI model failed #{attempt}"
                    )

                log.info(f"  AI Answer → CID={best_cid}")

                # ── Submit ──
                submit_url = data["shortlink_urls"][best_cid]
                log.info(f"  Submitting: {submit_url}")

                chain.append({
                    "captcha_num": attempt,
                    "choices":     choices,
                    "answer_cid":  best_cid,
                    "submit_url":  submit_url,
                })

                response = self.session.get(submit_url)
                if response is None:
                    return self._result(
                        False, chain,
                        f"Submit failed #{attempt}"
                    )

                Path(f"debug_page_{attempt}.html").write_text(
                    response.text, encoding="utf-8"
                )
                log.info(
                    f"  Response: "
                    f"{response.status_code} → {response.url}"
                )
                time.sleep(1.5)

        return self._result(
            False, chain, f"Exceeded {MAX_CAPTCHAS} limit"
        )

    def _result(self, success, chain, error="", bypassed=False):
        if not success:
            log.error(f"❌ {error}")
        return {
            "success":       success,
            "captchas_done": len(chain),
            "bypassed":      bypassed,
            "error":         error,
            "chain":         chain,
        }


# ─────────────────────────────────────────────
# LINK MANAGER
# ─────────────────────────────────────────────

class LinkManager:

    SAVE_FILE = Path("links.json")

    def __init__(self):
        self.links = self._load()

    def _load(self):
        if self.SAVE_FILE.exists():
            try:
                return json.loads(self.SAVE_FILE.read_text())
            except:
                return []
        return []

    def _save(self):
        self.SAVE_FILE.write_text(
            json.dumps(self.links, indent=2)
        )

    def _sanitize(self, url):
        return url.replace("&", "@@")

    def create_one(self):
        api = (
            f"{BASE_URL}/linkapi.php"
            f"?user={ZERADS_USER}"
            f"&url={self._sanitize(TARGET_URL)}"
            f"&furl={self._sanitize(FAIL_URL)}"
            f"&strict={STRICT}"
            f"&adsnum={ADS_NUM}"
        )
        try:
            r    = requests.get(api, timeout=15)
            code = r.text.strip()
            if not code or len(code) > 25 or " " in code:
                log.error(f"Bad API response: '{code}'")
                return None
            full = f"{BASE_URL}/{code}"
            self.links.append({
                "url":       full,
                "code":      code,
                "created":   time.strftime("%Y-%m-%d %H:%M:%S"),
                "uses":      0,
                "successes": 0,
                "active":    True,
            })
            self._save()
            log.info(f"✅ Created: {full}")
            return full
        except Exception as e:
            log.error(f"Create failed: {e}")
            return None

    def create_batch(self, n):
        urls = []
        for i in range(n):
            log.info(f"Creating link {i+1}/{n}...")
            url = self.create_one()
            if url:
                urls.append(url)
            time.sleep(1.5)
        return urls

    def get_active_urls(self):
        return [
            r["url"] for r in self.links
            if r.get("active", True)
        ]

    def ensure_minimum(self, minimum=LINKS_TO_KEEP):
        active = self.get_active_urls()
        needed = minimum - len(active)
        if needed > 0:
            log.info(f"Creating {needed} links...")
            self.create_batch(needed)
        return self.get_active_urls()

    def record(self, url, success):
        for r in self.links:
            if r["url"] == url:
                r["uses"] += 1
                if success:
                    r["successes"] += 1
                self._save()
                break


# ─────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────

class MainRunner:

    def __init__(self, use_tor=True):
        self.use_tor  = use_tor
        self.tor      = TorController()
        self.links    = LinkManager()
        self.analyser = PageAnalyser()
        self.ai       = AIModel(
            script    = MODEL_SCRIPT,
            image_dir = IMAGE_DIR,
        )
        self.stats = {
            "total_runs":      0,
            "succeeded":       0,
            "failed":          0,
            "bypassed":        0,
            "captchas_solved": 0,
            "run_times":       [],
        }

    def _new_solver(self):
        session = SessionManager(use_tor=self.use_tor)
        return ChainSolver(session, self.analyser, self.ai)

    def run(self, n, url=None):
        if not url:
            urls = self.links.ensure_minimum(LINKS_TO_KEEP)
        else:
            urls = [url]
        
        log.info(f"\n{'='*60}")
        log.info("PIPELINE RUNNER")
        log.info(f"  Runs  : {n}")
        log.info(f"  Links : {len(urls)}")
        log.info(f"  Tor   : {self.use_tor}")
        log.info(f"  Model : {MODEL_SCRIPT}")
        log.info(f"  ImgDir: {IMAGE_DIR}")
        log.info(f"{'='*60}")
        
        all_results = []
        # Track current IP — rotate only after destination hit
        ip = self.tor.rotate() if self.use_tor else "direct"
        log.info(f"  Initial IP: {ip}")
            
        for run_num in range(1, n + 1):
            t_start = time.time()
            
            log.info(f"\n{'\u2550'*60}")
            log.info(
                f"  RUN {run_num}/{n} | "
                f"\u2705 {self.stats['succeeded']} "
                f"\u274c {self.stats['failed']} | "
                f"\U0001f9e9 {self.stats['captchas_solved']} captchas"
            )
            log.info(f"{'\u2550'*60}")
            
            # Pick link
            current_url = (
                url if url
                else urls[(run_num - 1) % len(urls)]
            )
            log.info(f"  URL: {current_url}  |  IP: {ip}")
                
            # Solve — reuse current IP
            solver = self._new_solver()
            result = solver.solve(current_url)
            elapsed = round(time.time() - t_start, 1)
            
            self.stats["total_runs"] += 1
            self.stats["run_times"].append(elapsed)
            
            bypassed = result.get("bypassed", False)
            caps     = result.get("captchas_done", 0)
            
            if result["success"]:
                self.stats["succeeded"]      += 1
                self.stats["captchas_solved"] += caps
                if bypassed:
                    self.stats["bypassed"] += 1
                    log.warning(
                        f"  \u26a0\ufe0f  RUN {run_num} BYPASSED | "
                        f"0 earnings | {elapsed}s"
                    )
                else:
                    log.info(
                        f"  \u2705 RUN {run_num} SUCCESS | "
                        f"captchas={caps} | {elapsed}s"
                        
                    )
                    
                # ── Destination reached → NOW rotate IP ──
                if self.use_tor and run_num < n:
                    log.info("  Destination hit — rotating Tor IP...")
                    ip = self.tor.rotate()
                    log.info(f"  New IP: {ip}")
                        
            else:
                self.stats["failed"] += 1
                log.warning(
                    f"  \u274c RUN {run_num} FAILED | "
                    f"{result.get('error','?')} | {elapsed}s"
                    
                )
                # Failed — keep same IP, Zerads allows retry on same IP
            if not url:
                self.links.record(current_url, result["success"])
                
            result.update({
                "run_num"  : run_num,
                "ip_used"  : ip,
                "elapsed"  : elapsed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            all_results.append(result)

            Path("pipeline_results.json").write_text(
                json.dumps(all_results, indent=2)
            )

            avg = round(
                sum(self.stats["run_times"]) /
                len(self.stats["run_times"]), 1
            )

            eta = round((n - run_num) * avg / 60, 1)
            log.info(
                f"  📊 {run_num}/{n} | "
                f"captchas={self.stats['captchas_solved']} | "
                f"avg={avg}s | ETA≈{eta}min"
            )

            if run_num < n:
                time.sleep(BETWEEN_RUN_WAIT)

        # Final summary
        log.info(f"\n{'='*60}")
        log.info("FINAL SUMMARY")
        log.info(f"  ✅ Succeeded : {self.stats['succeeded']}")
        log.info(f"  ❌ Failed    : {self.stats['failed']}")
        log.info(f"  ⚠️  Bypassed  : {self.stats['bypassed']}")
        log.info(f"  🧩 Captchas  : {self.stats['captchas_solved']}")
        log.info(f"  🔄 Rotations : {self.tor.rotate_count}")
        log.info(f"{'='*60}")

        return all_results


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    args = {
        "n":           35,
        "links":       LINKS_TO_KEEP,
        "use_tor":     True,
        "url":         None,
        "create_only": False,
        "model":       str(MODEL_SCRIPT),
        "image_dir":   str(IMAGE_DIR),
    }
    argv = sys.argv[1:]
    i    = 0
    while i < len(argv):
        t = argv[i]
        if t == "--n":
            i += 1
            args["n"] = int(argv[i])
        elif t == "--links":
            i += 1
            args["links"] = int(argv[i])
        elif t == "--url":
            i += 1
            args["url"] = argv[i]
        elif t == "--no-tor":
            args["use_tor"] = False
        elif t == "--create-only":
            args["create_only"] = True
        elif t == "--model":
            i += 1
            args["model"] = argv[i]
        elif t == "--image-dir":
            i += 1
            args["image_dir"] = argv[i]
        elif t.startswith("http"):
            args["url"] = t
        i += 1
    return args


def main():
    print("""
╔══════════════════════════════════════════════╗
║  Zerads AI Pipeline Solver                  ║
║  ONNX Model + Tor Rotation                  ║
║  File-based image integration               ║
╚══════════════════════════════════════════════╝
""")
    args = parse_args()

    # Override paths from args
    global MODEL_SCRIPT, IMAGE_DIR
    MODEL_SCRIPT = Path(args["model"])
    IMAGE_DIR    = Path(args["image_dir"])

    if args["create_only"]:
        lm = LinkManager()
        lm.create_batch(args["links"])
        return

    runner = MainRunner(use_tor=args["use_tor"])
    runner.run(n=args["n"], url=args["url"])


if __name__ == "__main__":
    main()
