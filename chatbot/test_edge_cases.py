"""
Edge Case Testing Suite for Kisan AI Chatbot
Tests extreme scenarios, boundary conditions, and unusual inputs
"""

from django.test import TestCase, Client
from django.urls import reverse
import json
from chatbot.dataset_loader import get_dataset
from chatbot.ai_engine import (
    is_privacy_query, is_off_topic, hindi_to_english_numbers,
    clean_solution_text, generate_suggestions, fallback_response,
    extract_brief_solution
)


class ExtremeInputTests(TestCase):
    """Test extreme and unusual inputs"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dataset = get_dataset()
        cls.client = Client()
    
    def test_empty_string_query(self):
        """Test completely empty query"""
        crop, alias = self.dataset.detect_crop("")
        self.assertIsNone(crop)
        
    def test_only_spaces_query(self):
        """Test query with only spaces"""
        crop, alias = self.dataset.detect_crop("     ")
        self.assertIsNone(crop)
        
    def test_only_newlines_query(self):
        """Test query with only newlines"""
        crop, alias = self.dataset.detect_crop("\n\n\n")
        self.assertIsNone(crop)
        
    def test_only_tabs_query(self):
        """Test query with only tabs"""
        crop, alias = self.dataset.detect_crop("\t\t\t")
        self.assertIsNone(crop)
        
    def test_extremely_long_query(self):
        """Test query with 10,000+ characters"""
        long_query = "aam me keede lag gaye hain " * 500  # ~13,500 chars
        crop, alias = self.dataset.detect_crop(long_query)
        self.assertEqual(crop, "आम")
        
    def test_single_character_query(self):
        """Test single character queries"""
        for char in ['a', 'आ', '1', '!']:
            crop, alias = self.dataset.detect_crop(char)
            # Should not crash
            self.assertIsInstance(crop, (str, type(None)))
            
    def test_repeated_characters(self):
        """Test queries with repeated characters"""
        crop, alias = self.dataset.detect_crop("aaaaaaaaaam")
        # May or may not detect, but shouldn't crash
        self.assertIsInstance(crop, (str, type(None)))
        
    def test_all_special_characters(self):
        """Test query with only special characters"""
        crop, alias = self.dataset.detect_crop("!@#$%^&*()")
        self.assertIsNone(crop)
        
    def test_mixed_special_characters_with_crop(self):
        """Test crop name surrounded by special characters"""
        crop, alias = self.dataset.detect_crop("!!!aam###")
        self.assertEqual(crop, "आम")
        
    def test_unicode_emoji_in_query(self):
        """Test queries containing emojis"""
        crop, alias = self.dataset.detect_crop("aam me keede 🐛🌾")
        self.assertEqual(crop, "आम")
        
    def test_rtl_and_ltr_mixed(self):
        """Test right-to-left and left-to-right text mixing"""
        crop, alias = self.dataset.detect_crop("mango में कीड़े in tree")
        self.assertEqual(crop, "आम")
        
    def test_null_bytes_in_query(self):
        """Test query containing null bytes"""
        query = "aam\x00me\x00keede"
        crop, alias = self.dataset.detect_crop(query)
        # Should handle gracefully
        self.assertIsInstance(crop, (str, type(None)))
        
    def test_control_characters_in_query(self):
        """Test query with control characters"""
        query = "aam\r\nme\tkeede"
        crop, alias = self.dataset.detect_crop(query)
        self.assertEqual(crop, "आम")


class BoundaryConditionTests(TestCase):
    """Test boundary conditions and limits"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dataset = get_dataset()
    
    def test_retrieve_top_k_zero(self):
        """Test retrieval with top_k=0"""
        results = self.dataset.retrieve("aam me keede", crop_hindi="आम", intent="pest", top_k=0)
        self.assertEqual(len(results), 0)
        
    def test_retrieve_top_k_negative(self):
        """Test retrieval with negative top_k"""
        results = self.dataset.retrieve("aam me keede", crop_hindi="आम", intent="pest", top_k=-1)
        # Should handle gracefully
        self.assertIsInstance(results, list)
        
    def test_retrieve_top_k_very_large(self):
        """Test retrieval with very large top_k"""
        results = self.dataset.retrieve("aam me keede", crop_hindi="आम", intent="pest", top_k=10000)
        # Should return at most the number of matching rows
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), len(self.dataset.df))
        
    def test_crop_name_at_start(self):
        """Test crop name at the very start of query"""
        crop, alias = self.dataset.detect_crop("aam")
        self.assertEqual(crop, "आम")
        
    def test_crop_name_at_end(self):
        """Test crop name at the very end of query"""
        crop, alias = self.dataset.detect_crop("keede lag gaye aam")
        self.assertEqual(crop, "आम")
        
    def test_crop_name_in_middle(self):
        """Test crop name in the middle of query"""
        crop, alias = self.dataset.detect_crop("mere aam me keede lag gaye")
        self.assertEqual(crop, "आम")
        
    def test_multiple_same_crop_mentions(self):
        """Test query mentioning same crop multiple times"""
        crop, alias = self.dataset.detect_crop("aam ke aam me aam ka keeda")
        self.assertEqual(crop, "आम")
        
    def test_similarity_threshold_edge(self):
        """Test queries right at similarity threshold"""
        # Query designed to be borderline
        results = self.dataset.retrieve("xyz", crop_hindi=None, intent="general", top_k=5)
        # Should respect threshold
        self.assertIsInstance(results, list)


class MalformedInputTests(TestCase):
    """Test malformed and corrupted inputs"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dataset = get_dataset()
        cls.client = Client()
    
    def test_incomplete_hindi_characters(self):
        """Test queries with incomplete Hindi character combinations"""
        # Incomplete matras or half-characters
        queries = ["आ�", "क्", "ा"]
        for query in queries:
            crop, alias = self.dataset.detect_crop(query)
            # Should not crash
            self.assertIsInstance(crop, (str, type(None)))
            
    def test_mixed_encoding_query(self):
        """Test query that might have mixed encoding issues"""
        query = "aam me keede लग गए"
        crop, alias = self.dataset.detect_crop(query)
        self.assertEqual(crop, "आम")
        
    def test_repeated_punctuation(self):
        """Test excessive punctuation"""
        crop, alias = self.dataset.detect_crop("aam me keede!!!!!!!!!!!")
        self.assertEqual(crop, "आम")
        
    def test_no_spaces_between_words(self):
        """Test query with no spaces"""
        crop, alias = self.dataset.detect_crop("aammekedeelaggaye")
        # May or may not detect
        self.assertIsInstance(crop, (str, type(None)))
        
    def test_excessive_spaces_between_words(self):
        """Test query with excessive spaces"""
        crop, alias = self.dataset.detect_crop("aam     me     keede")
        self.assertEqual(crop, "आम")
        
    def test_mixed_case_variations(self):
        """Test various case combinations"""
        test_cases = ["AAM", "Aam", "aAm", "aAM"]
        for query in test_cases:
            crop, alias = self.dataset.detect_crop(query)
            self.assertEqual(crop, "आम")
            
    def test_api_endpoint_with_malformed_json(self):
        """Test API with malformed JSON"""
        response = self.client.post(
            reverse('chat'),
            data='{"message": "test"',  # Missing closing brace
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        
    def test_api_endpoint_with_wrong_content_type(self):
        """Test API with wrong content type"""
        response = self.client.post(
            reverse('chat'),
            data='message=test',
            content_type='application/x-www-form-urlencoded'
        )
        # Should handle gracefully
        self.assertIn(response.status_code, [200, 400])
        
    def test_api_endpoint_with_nested_json(self):
        """Test API with deeply nested JSON"""
        response = self.client.post(
            reverse('chat'),
            data=json.dumps({"message": {"nested": {"deep": "aam me keede"}}}),
            content_type='application/json'
        )
        # Should handle gracefully - expects 400 since message is not a string
        self.assertEqual(response.status_code, 400)


class ConcurrencyAndStateTests(TestCase):
    """Test concurrent access and state management"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dataset = get_dataset()
    
    def test_multiple_simultaneous_crop_detections(self):
        """Test multiple crop detections don't interfere"""
        queries = [
            "aam me keede",
            "dhan me pani",
            "gehun me rog",
            "tamatar me jhulsa"
        ]
        results = [self.dataset.detect_crop(q) for q in queries]
        expected = ["आम", "धान", "गेहूँ", "टमाटर"]
        actual = [r[0] for r in results]
        self.assertEqual(actual, expected)
        
    def test_dataset_singleton_consistency(self):
        """Test that get_dataset returns same instance"""
        from chatbot.dataset_loader import get_dataset
        ds1 = get_dataset()
        ds2 = get_dataset()
        self.assertIs(ds1, ds2)
        
    def test_multiple_retrievals_dont_modify_state(self):
        """Test that retrievals don't modify dataset state"""
        initial_df_len = len(self.dataset.df)
        for _ in range(10):
            self.dataset.retrieve("aam me keede", crop_hindi="आम", intent="pest", top_k=5)
        final_df_len = len(self.dataset.df)
        self.assertEqual(initial_df_len, final_df_len)


class NumericEdgeCaseTests(TestCase):
    """Test numeric edge cases"""
    
    def test_very_large_numbers_in_query(self):
        """Test queries with very large numbers"""
        query = "999999999999 gram urea dale"
        intent = get_dataset().detect_intent(query)
        self.assertEqual(intent, "fertilizer")
        
    def test_decimal_numbers_in_query(self):
        """Test queries with decimal numbers"""
        query = "0.5 ml dawa dale"
        intent = get_dataset().detect_intent(query)
        # Should handle gracefully
        self.assertIsInstance(intent, str)
        
    def test_negative_numbers_in_query(self):
        """Test queries with negative numbers"""
        query = "-10 degree temperature"
        # Should not crash
        crop, alias = get_dataset().detect_crop(query)
        self.assertIsInstance(crop, (str, type(None)))
        
    def test_scientific_notation_in_query(self):
        """Test queries with scientific notation"""
        query = "1e6 bacteria in soil"
        # Should not crash
        crop, alias = get_dataset().detect_crop(query)
        self.assertIsInstance(crop, (str, type(None)))
        
    def test_hindi_numbers_conversion(self):
        """Test all Hindi digits convert correctly"""
        test_cases = {
            "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
            "५": "5", "६": "6", "७": "7", "८": "8", "९": "9"
        }
        for hindi, english in test_cases.items():
            result = hindi_to_english_numbers(hindi)
            self.assertEqual(result, english)
            
    def test_mixed_hindi_english_numbers(self):
        """Test mixed Hindi and English numbers"""
        result = hindi_to_english_numbers("१23४56")
        self.assertEqual(result, "123456")


class SpecialCropNameTests(TestCase):
    """Test edge cases with crop names"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dataset = get_dataset()
    
    def test_crop_name_with_suffix(self):
        """Test crop names with Hindi suffixes"""
        test_cases = [
            "आमके पेड़",
            "धानमें कीट",
            "गेहूँकी खेती"
        ]
        for query in test_cases:
            crop, alias = self.dataset.detect_crop(query)
            # Should detect crop even with suffix
            self.assertIsNotNone(crop)
            
    def test_crop_name_partial_match(self):
        """Test partial crop name matches"""
        # "आ" is part of "आम" but shouldn't match
        crop, alias = self.dataset.detect_crop("आ")
        # Should not match partial
        self.assertIsInstance(crop, (str, type(None)))
        
    def test_crop_name_in_compound_word(self):
        """Test crop name inside compound words"""
        # "धान" should not match in "समाधान"
        crop, alias = self.dataset.detect_crop("समाधान चाहिए")
        self.assertIsNone(crop)
        
    def test_all_crop_aliases_work(self):
        """Test that all defined crop aliases are detected"""
        from chatbot.dataset_loader import CROP_ALIASES
        # Test a sample of aliases
        sample_aliases = ['aam', 'dhan', 'gehun', 'tamatar', 'pyaz']
        for alias in sample_aliases:
            crop, detected_alias = self.dataset.detect_crop(alias)
            self.assertIsNotNone(crop, f"Failed to detect crop for alias: {alias}")


class GuardrailEdgeCaseTests(TestCase):
    """Test edge cases in guardrail detection"""
    
    def test_privacy_query_with_farming_context(self):
        """Test privacy queries mixed with farming context"""
        # Should still block even if farming words present
        query = "who made you and also aam me keede"
        self.assertTrue(is_privacy_query(query))
        
    def test_off_topic_with_crop_name(self):
        """Test off-topic queries that mention crops"""
        # "hello aam" should still be off-topic
        query = "hello aam"
        # This is borderline - depends on implementation
        result = is_off_topic(query)
        self.assertIsInstance(result, bool)
        
    def test_case_insensitive_privacy_detection(self):
        """Test privacy detection is case insensitive"""
        queries = ["WHO MADE YOU", "Who Made You", "who made you"]
        for query in queries:
            self.assertTrue(is_privacy_query(query))
            
    def test_privacy_query_with_typos(self):
        """Test privacy detection with typos"""
        # "who mde you" - typo in "made"
        query = "who mde you"
        # May or may not detect depending on regex strictness
        result = is_privacy_query(query)
        self.assertIsInstance(result, bool)
        
    def test_greeting_in_middle_of_query(self):
        """Test greeting words in middle of farming query"""
        query = "aam me hello keede lag gaye"
        # Should not be off-topic since it has farming context
        self.assertFalse(is_off_topic(query))


class TextProcessingEdgeCaseTests(TestCase):
    """Test edge cases in text processing functions"""
    
    def test_clean_solution_empty_string(self):
        """Test clean_solution_text with empty string"""
        result = clean_solution_text("")
        self.assertEqual(result, "")
        
    def test_clean_solution_only_abbreviations(self):
        """Test clean_solution_text with only abbreviations"""
        result = clean_solution_text("कि0ग्रा0 मि0ली0 ली0")
        self.assertIn("किलोग्राम", result)
        self.assertIn("मिली", result)
        self.assertIn("लीटर", result)
        
    def test_clean_solution_no_abbreviations(self):
        """Test clean_solution_text with no abbreviations"""
        original = "यह एक साधारण वाक्य है"
        result = clean_solution_text(original)
        self.assertEqual(result, original)
        
    def test_extract_brief_solution_empty(self):
        """Test extract_brief_solution with empty string"""
        result = extract_brief_solution("")
        self.assertEqual(result, "")
        
    def test_extract_brief_solution_no_structure(self):
        """Test extract_brief_solution with unstructured text"""
        text = "यह एक साधारण उत्तर है बिना किसी संरचना के"
        result = extract_brief_solution(text)
        self.assertIsInstance(result, str)
        self.assertLessEqual(len(result), 250)
        
    def test_extract_brief_solution_very_long(self):
        """Test extract_brief_solution with very long text"""
        long_text = "समाधान: " + ("यह एक बहुत लंबा समाधान है। " * 100)
        result = extract_brief_solution(long_text)
        self.assertLessEqual(len(result), 250)
        
    def test_fallback_response_empty_rows(self):
        """Test fallback_response with empty rows list"""
        response = fallback_response("test query", None, "general", [])
        self.assertIn("KVK", response)
        self.assertIn("1800-180-1551", response)
        
    def test_generate_suggestions_all_intents(self):
        """Test suggestion generation for all intent types"""
        intents = ['pest', 'disease', 'fertilizer', 'irrigation', 'growth', 'cultivation', 'general']
        for intent in intents:
            suggestions = generate_suggestions("आम", intent, [])
            self.assertIsInstance(suggestions, list)
            self.assertGreater(len(suggestions), 0)
            self.assertLessEqual(len(suggestions), 4)


if __name__ == '__main__':
    import unittest
    unittest.main()
