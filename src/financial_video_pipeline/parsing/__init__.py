"""Content parsing module for converting JSON to structured sections and images."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ..config import Settings
from ..utils import ensure_directory, slugify, append_to_log, load_processed_set


logger = logging.getLogger(__name__)


class ContentParser:
    """Parses JSON content into structured sections and downloads images."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with default headers."""
        session = requests.Session()
        session.headers.update({
            "User-Agent": self.settings.parsing.user_agent,
            "Accept": "*/*",
            "Referer": "https://www.tradingview.com/",
        })
        return session
    
    def process_file(self, json_path: Path, week_id: Optional[str] = None) -> bool:
        """
        Process a JSON file into structured sections and images.
        
        Args:
            json_path: Path to JSON file to process
            week_id: Week identifier. If None, extracted from filename.
            
        Returns:
            True if processing succeeded, False otherwise
        """
        if not json_path.exists():
            logger.error(f"JSON file not found: {json_path}")
            return False
        
        if week_id is None:
            from ..utils import extract_week_id
            week_id = extract_week_id(json_path.name)
            if not week_id:
                logger.error(f"Could not extract week ID from filename: {json_path.name}")
                return False
        
        # Check if already processed
        log_file = self._get_log_file()
        processed = load_processed_set(log_file)
        if json_path.name in processed:
            logger.info(f"Skipping already processed file: {json_path.name}")
            return True
        
        try:
            # Load and validate JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not self._validate_json_structure(data):
                logger.error(f"Invalid JSON structure in {json_path.name}")
                return False
            
            # Setup output directories
            output_dir = self._get_output_dir(week_id)
            sections_dir = output_dir / self.settings.parsing.sections_subdir
            images_dir = output_dir / self.settings.parsing.images_subdir
            
            ensure_directory(sections_dir)
            ensure_directory(images_dir)
            
            # Process sections
            success = self._process_sections(data["sections"], sections_dir, images_dir)
            
            if success:
                append_to_log(log_file, json_path.name)
                logger.info(f"Successfully processed {json_path.name} -> {output_dir.relative_to(self.settings.project_root)}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing {json_path.name}: {e}")
            return False
    
    def _validate_json_structure(self, data: dict) -> bool:
        """Validate JSON has required structure."""
        if not isinstance(data, dict):
            return False
        
        if "sections" not in data:
            return False
        
        if not isinstance(data["sections"], list):
            return False
        
        return True
    
    def _process_sections(self, sections: List[dict], sections_dir: Path, images_dir: Path) -> bool:
        """Process list of sections into text files and images."""
        total_written = 0
        
        for idx, section in enumerate(sections, start=1):
            if not isinstance(section, dict):
                logger.warning(f"Skipping invalid section at index {idx}")
                continue
            
            title = section.get("title") or ""
            if title:
                title = title.strip()
            slug = section.get("slug") or ""
            if slug:
                slug = slug.strip()
            slug = slug or slugify(title or f"section-{idx}")
            script = section.get("script", "")
            image_url = section.get("image_url") or ""
            if image_url:
                image_url = image_url.strip()
            
            # Write script file
            script_name = f"{idx:02d}_{slug}.txt"
            script_path = sections_dir / script_name
            
            try:
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(script)
                total_written += 1
                logger.debug(f"Wrote script: {script_name}")
            except Exception as e:
                logger.error(f"Error writing script {script_name}: {e}")
                continue
            
            # Download image if provided
            if image_url:
                image_base = images_dir / f"{idx:02d}_{slug}"
                self._download_image(image_url, image_base)
        
        logger.info(f"Processed {total_written} sections")
        return total_written > 0
    
    def _download_image(self, url: str, dest_base: Path) -> Optional[Path]:
        """Download image from URL to destination."""
        try:
            # Try direct image download first
            image_path = self._try_direct_image_download(url, dest_base)
            if image_path:
                return image_path
            
            # Try HTML scraping if enabled
            if self.settings.parsing.enable_html_scrape:
                image_path = self._try_html_image_scraping(url, dest_base)
                if image_path:
                    return image_path
            
            # Fall back to writing URL as .link file
            link_file = dest_base.with_suffix(".link")
            with open(link_file, 'w', encoding='utf-8') as f:
                f.write(f"{url}\n")
            logger.debug(f"Saved URL as link file: {link_file.name}")
            return link_file
            
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return None
    
    def _try_direct_image_download(self, url: str, dest_base: Path) -> Optional[Path]:
        """Try to download URL as direct image."""
        try:
            response = self._fetch_with_retries(url, stream=True)
            content_type = response.headers.get("Content-Type", "").lower()
            
            if content_type.startswith("image/"):
                ext = self._guess_extension_from_content_type(content_type)
                output_path = dest_base.with_suffix(ext)
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                logger.debug(f"Downloaded direct image: {output_path.name}")
                return output_path
            
        except Exception:
            pass
        
        return None
    
    def _try_html_image_scraping(self, url: str, dest_base: Path) -> Optional[Path]:
        """Try to extract image URL from HTML page and download it."""
        try:
            response = self._fetch_with_retries(url)
            content_type = response.headers.get("Content-Type", "").lower()
            
            # Check if response looks like HTML
            if not ("text/html" in content_type or "application/xhtml" in content_type):
                return None
            
            # Parse HTML and extract image URL
            image_url = self._extract_image_from_html(response.text, url)
            if not image_url:
                return None
            
            # Download the extracted image
            return self._try_direct_image_download(image_url, dest_base)
            
        except Exception:
            pass
        
        return None
    
    def _extract_image_from_html(self, html_text: str, base_url: str) -> Optional[str]:
        """Extract image URL from HTML using BeautifulSoup."""
        try:
            soup = BeautifulSoup(html_text, "html.parser")
            
            # Try OpenGraph image
            og_image = soup.find("meta", property="og:image")
            if og_image and og_image.get("content"):
                return urljoin(base_url, og_image["content"])
            
            # Try Twitter image
            twitter_image = soup.find("meta", attrs={"name": "twitter:image"})
            if twitter_image and twitter_image.get("content"):
                return urljoin(base_url, twitter_image["content"])
            
            # Try TradingView snapshot image
            tv_image = soup.find("img", class_=lambda x: x and "tv-snapshot-image" in x)
            if tv_image and tv_image.get("src"):
                return urljoin(base_url, tv_image["src"])
            
            # Fallback to first image
            first_img = soup.find("img")
            if first_img and first_img.get("src"):
                return urljoin(base_url, first_img["src"])
            
        except Exception as e:
            logger.debug(f"Error parsing HTML: {e}")
        
        return None
    
    def _fetch_with_retries(self, url: str, stream: bool = False) -> requests.Response:
        """Fetch URL with retries."""
        last_exception = None
        
        for attempt in range(self.settings.parsing.retries + 1):
            try:
                response = self.session.get(
                    url, 
                    timeout=self.settings.parsing.timeout_sec,
                    allow_redirects=True,
                    stream=stream
                )
                response.raise_for_status()
                return response
                
            except Exception as e:
                last_exception = e
                if attempt < self.settings.parsing.retries:
                    time.sleep(0.5 * (attempt + 1))
        
        raise last_exception or Exception("All retries failed")
    
    def _guess_extension_from_content_type(self, content_type: str) -> str:
        """Guess file extension from Content-Type header."""
        content_type = content_type.lower().split(";")[0].strip()
        
        mapping = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/svg+xml": ".svg",
        }
        
        return mapping.get(content_type, ".png")
    
    def _get_output_dir(self, week_id: str) -> Path:
        """Get output directory for a specific week."""
        return self.settings.get_absolute_path(self.settings.paths.parsed_dir) / week_id
    
    def _get_log_file(self) -> Path:
        """Get path to processing log file."""
        return self.settings.get_absolute_path(self.settings.paths.parsed_dir) / "parsing.log"
    
    def list_available_weeks(self) -> List[str]:
        """List all available weeks in parsed directory."""
        parsed_dir = self.settings.get_absolute_path(self.settings.paths.parsed_dir)
        if not parsed_dir.exists():
            return []
        
        weeks = []
        for item in parsed_dir.iterdir():
            if item.is_dir() and item.name != "__pycache__":
                weeks.append(item.name)
        
        return sorted(weeks)