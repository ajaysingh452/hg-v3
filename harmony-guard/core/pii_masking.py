"""PII (Personally Identifiable Information) masking for privacy compliance."""

import re
from typing import Dict, List, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


class PIIMasker:
    """Handles detection and masking of personally identifiable information."""
    
    def __init__(self, config: Dict):
        """Initialize PII masker with configuration."""
        self.config = config
        self.enabled = config.get("enabled", False)
        self.mask_emails = config.get("mask_emails", True)
        self.mask_phones = config.get("mask_phones", True)
        self.mask_ids = config.get("mask_ids", True)
        self.mask_character = config.get("mask_character", "*")
        
        # Compile regex patterns for different PII types
        self._compile_patterns()
    
    def mask_pii(self, text: str) -> Tuple[str, bool, Dict[str, List[str]]]:
        """
        Mask PII in text and return masked text with detection info.
        
        Args:
            text: Input text to process
            
        Returns:
            Tuple of (masked_text, pii_found, detected_pii_types)
        """
        if not self.enabled:
            return text, False, {}
        
        masked_text = text
        pii_found = False
        detected_pii = {}
        
        # Mask emails
        if self.mask_emails:
            masked_text, email_found, emails = self._mask_emails(masked_text)
            if email_found:
                pii_found = True
                detected_pii["emails"] = emails
        
        # Mask phone numbers
        if self.mask_phones:
            masked_text, phone_found, phones = self._mask_phone_numbers(masked_text)
            if phone_found:
                pii_found = True
                detected_pii["phones"] = phones
        
        # Mask ID numbers
        if self.mask_ids:
            masked_text, id_found, ids = self._mask_id_numbers(masked_text)
            if id_found:
                pii_found = True
                detected_pii["ids"] = ids
        
        # Mask credit card numbers
        masked_text, cc_found, cards = self._mask_credit_cards(masked_text)
        if cc_found:
            pii_found = True
            detected_pii["credit_cards"] = cards
        
        # Mask IP addresses
        masked_text, ip_found, ips = self._mask_ip_addresses(masked_text)
        if ip_found:
            pii_found = True
            detected_pii["ip_addresses"] = ips
        
        return masked_text, pii_found, detected_pii
    
    def detect_pii_types(self, text: str) -> List[str]:
        """
        Detect types of PII present in text without masking.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected PII types
        """
        pii_types = []
        
        if self._has_emails(text):
            pii_types.append("email")
        
        if self._has_phone_numbers(text):
            pii_types.append("phone")
        
        if self._has_id_numbers(text):
            pii_types.append("id_number")
        
        if self._has_credit_cards(text):
            pii_types.append("credit_card")
        
        if self._has_ip_addresses(text):
            pii_types.append("ip_address")
        
        return pii_types
    
    def _mask_emails(self, text: str) -> Tuple[str, bool, List[str]]:
        """Mask email addresses in text."""
        emails = []
        email_pattern = self.email_pattern
        
        def replace_email(match):
            email = match.group(0)
            emails.append(email)
            return "[EMAIL]"
        
        masked_text = email_pattern.sub(replace_email, text)
        return masked_text, len(emails) > 0, emails
    
    def _mask_phone_numbers(self, text: str) -> Tuple[str, bool, List[str]]:
        """Mask phone numbers in text."""
        phones = []
        
        # Multiple phone patterns for different formats
        patterns = [
            self.phone_pattern_1,  # +91-9876543210
            self.phone_pattern_2,  # (123) 456-7890
            self.phone_pattern_3,  # 123-456-7890
            self.phone_pattern_4,  # 9876543210 (10 digits)
        ]
        
        masked_text = text
        
        for pattern in patterns:
            def replace_phone(match):
                phone = match.group(0)
                phones.append(phone)
                return "[PHONE]"
            
            masked_text = pattern.sub(replace_phone, masked_text)
        
        return masked_text, len(phones) > 0, phones
    
    def _mask_id_numbers(self, text: str) -> Tuple[str, bool, List[str]]:
        """Mask ID numbers (Aadhaar, PAN, etc.) in text."""
        ids = []
        
        # Indian ID patterns
        patterns = [
            self.aadhaar_pattern,  # Aadhaar: 1234 5678 9012
            self.pan_pattern,      # PAN: ABCDE1234F
            self.ssn_pattern,      # SSN: 123-45-6789
        ]
        
        masked_text = text
        
        for pattern in patterns:
            def replace_id(match):
                id_num = match.group(0)
                ids.append(id_num)
                return "[ID]"
            
            masked_text = pattern.sub(replace_id, masked_text)
        
        return masked_text, len(ids) > 0, ids
    
    def _mask_credit_cards(self, text: str) -> Tuple[str, bool, List[str]]:
        """Mask credit card numbers in text."""
        cards = []
        
        def replace_card(match):
            card = match.group(0)
            cards.append(card)
            return "[CARD]"
        
        masked_text = self.credit_card_pattern.sub(replace_card, text)
        return masked_text, len(cards) > 0, cards
    
    def _mask_ip_addresses(self, text: str) -> Tuple[str, bool, List[str]]:
        """Mask IP addresses in text."""
        ips = []
        
        def replace_ip(match):
            ip = match.group(0)
            ips.append(ip)
            return "[IP]"
        
        masked_text = self.ip_pattern.sub(replace_ip, text)
        return masked_text, len(ips) > 0, ips
    
    def _has_emails(self, text: str) -> bool:
        """Check if text contains email addresses."""
        return bool(self.email_pattern.search(text))
    
    def _has_phone_numbers(self, text: str) -> bool:
        """Check if text contains phone numbers."""
        patterns = [
            self.phone_pattern_1,
            self.phone_pattern_2,
            self.phone_pattern_3,
            self.phone_pattern_4,
        ]
        return any(pattern.search(text) for pattern in patterns)
    
    def _has_id_numbers(self, text: str) -> bool:
        """Check if text contains ID numbers."""
        patterns = [
            self.aadhaar_pattern,
            self.pan_pattern,
            self.ssn_pattern,
        ]
        return any(pattern.search(text) for pattern in patterns)
    
    def _has_credit_cards(self, text: str) -> bool:
        """Check if text contains credit card numbers."""
        return bool(self.credit_card_pattern.search(text))
    
    def _has_ip_addresses(self, text: str) -> bool:
        """Check if text contains IP addresses."""
        return bool(self.ip_pattern.search(text))
    
    def _compile_patterns(self):
        """Compile regex patterns for PII detection."""
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        )
        
        # Phone number patterns
        # Pattern 1: +91-9876543210 or +1-123-456-7890
        self.phone_pattern_1 = re.compile(
            r'\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
        )
        
        # Pattern 2: (123) 456-7890
        self.phone_pattern_2 = re.compile(
            r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}'
        )
        
        # Pattern 3: 123-456-7890 or 123.456.7890
        self.phone_pattern_3 = re.compile(
            r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
        )
        
        # Pattern 4: 10-digit number (Indian mobile)
        self.phone_pattern_4 = re.compile(
            r'\b[6-9]\d{9}\b'
        )
        
        # Aadhaar pattern: 1234 5678 9012 or 123456789012
        self.aadhaar_pattern = re.compile(
            r'\b\d{4}\s?\d{4}\s?\d{4}\b'
        )
        
        # PAN pattern: ABCDE1234F
        self.pan_pattern = re.compile(
            r'\b[A-Z]{5}\d{4}[A-Z]\b'
        )
        
        # SSN pattern: 123-45-6789
        self.ssn_pattern = re.compile(
            r'\b\d{3}-\d{2}-\d{4}\b'
        )
        
        # Credit card pattern: 4-digit groups
        self.credit_card_pattern = re.compile(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        )
        
        # IP address pattern: IPv4
        self.ip_pattern = re.compile(
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        )
    
    def create_masked_logging_text(self, text: str) -> str:
        """
        Create a version of text safe for logging with PII masked.
        
        Args:
            text: Original text
            
        Returns:
            Text with PII masked for safe logging
        """
        if not self.enabled:
            return text
        
        masked_text, _, _ = self.mask_pii(text)
        return masked_text
    
    def get_pii_summary(self, text: str) -> Dict[str, int]:
        """
        Get summary of PII types and counts in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with PII type counts
        """
        summary = {
            "emails": 0,
            "phones": 0,
            "ids": 0,
            "credit_cards": 0,
            "ip_addresses": 0
        }
        
        if not self.enabled:
            return summary
        
        # Count emails
        summary["emails"] = len(self.email_pattern.findall(text))
        
        # Count phone numbers
        phone_patterns = [
            self.phone_pattern_1,
            self.phone_pattern_2,
            self.phone_pattern_3,
            self.phone_pattern_4,
        ]
        for pattern in phone_patterns:
            summary["phones"] += len(pattern.findall(text))
        
        # Count ID numbers
        id_patterns = [
            self.aadhaar_pattern,
            self.pan_pattern,
            self.ssn_pattern,
        ]
        for pattern in id_patterns:
            summary["ids"] += len(pattern.findall(text))
        
        # Count credit cards
        summary["credit_cards"] = len(self.credit_card_pattern.findall(text))
        
        # Count IP addresses
        summary["ip_addresses"] = len(self.ip_pattern.findall(text))
        
        return summary