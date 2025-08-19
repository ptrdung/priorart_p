"""
Patent Information Crawler
Module for extracting patent information from Google Patents and other sources
"""

import requests
from bs4 import BeautifulSoup
import time
from typing import Dict, Optional

class PatentCrawler:
    """Crawler for extracting patent information from web sources"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    
    def extract_patent_info(self, url: str) -> Dict[str, str]:
        """
        Extract patent information from a Google Patents URL.
        
        Args:
            url: Patent URL (e.g., Google Patents link)
            
        Returns:
            Dictionary containing title, abstract, claims, and description
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            error_message = f"HTTP error accessing {url}: {http_err}"
            return self._create_error_response(error_message)
        except requests.exceptions.Timeout:
            error_message = f"Request to {url} timed out."
            return self._create_error_response(error_message)
        except requests.exceptions.RequestException as err:
            error_message = f"Request error to {url}: {err}"
            return self._create_error_response(error_message)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        return {
            "title": self._extract_title(soup),
            "abstract": self._extract_abstract(soup),
            "claims": self._extract_claims(soup),
            "description": self._extract_description(soup)
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, str]:
        """Create error response dictionary"""
        return {
            "title": error_message,
            "abstract": error_message,
            "claims": error_message,
            "description": error_message
        }
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract patent title"""
        title_element = soup.find("title")
        if title_element:
            return title_element.get_text(strip=True)
        return "Title not found."
    
    def _extract_abstract(self, soup: BeautifulSoup) -> str:
        """Extract patent abstract"""
        abstract_element = soup.find(attrs={'itemprop': 'abstract'})
        if abstract_element:
            # Remove unwanted elements
            for bad in abstract_element.select("aside, .google-src-text, h2"):
                bad.decompose()
            return abstract_element.get_text(separator=' ', strip=True)
        return "Abstract not found."
    
    def _extract_claims(self, soup: BeautifulSoup) -> str:
        """Extract patent claims"""
        claims_list = []
        claims_section = soup.find('section', attrs={'itemprop': 'claims'})
        
        if claims_section:
            # Try first method
            for claim_div in claims_section.find_all('div', class_='claim-text'):
                text = ' '.join(claim_div.get_text(separator=' ', strip=True).split())
                if text:
                    claims_list.append(text)
            
            # Try alternative method if first fails
            if not claims_list:
                for claim_tag_element in claims_section.find_all('claim-text'):
                    for bad in claim_tag_element.select(".google-src-text"):
                        bad.decompose()
                    text = ' '.join(claim_tag_element.get_text(separator=' ', strip=True).split())
                    if text:
                        claims_list.append(text)
        
        if claims_list:
            return " ".join(claims_list)
        return f"Claims not found for the patent. Please check the page structure."
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract patent description"""
        description_list = []
        description_section = soup.find('section', attrs={'itemprop': 'description'})
        
        if description_section:
            # Try first method
            for description_div in description_section.find_all('div', class_='description-paragraph'):
                text = ' '.join(description_div.get_text(separator=' ', strip=True).split())
                if text:
                    description_list.append(text)
            
            # Try alternative method if first fails
            if not description_list:
                for description_tag_element in description_section.find_all('span', class_='notranslate'):
                    for bad in description_tag_element.select(".google-src-text"):
                        bad.decompose()
                    text = ' '.join(description_tag_element.get_text(separator=' ', strip=True).split())
                    if text:
                        description_list.append(text)
        
        if description_list:
            return " ".join(description_list)
        return "Description not found!"

# Legacy function for backward compatibility
def lay_thong_tin_patent(url: str) -> Dict[str, str]:
    """
    Legacy function wrapper for patent information extraction.
    
    Args:
        url: Patent URL
        
    Returns:
        Dictionary with patent information
    """
    crawler = PatentCrawler()
    return crawler.extract_patent_info(url)
