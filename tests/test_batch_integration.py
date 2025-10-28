"""
Integration tests for evals/batch.py with real OpenAI API calls.

These tests require a valid OPENAI_API_KEY environment variable.
They will make actual API calls and may incur costs.

To run these tests:
    OPENAI_API_KEY=your_key_here python -m pytest tests/test_batch_integration.py -v

Or skip them in regular test runs:
    python -m pytest tests/ -k "not integration"
"""

import os
import unittest
import time
from unittest.mock import Mock
from BetterTherapy.protocol import InferenceSynapse
from evals.batch import OpenAIBatchLLMAsJudgeEval
import dotenv
dotenv.load_dotenv()


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY environment variable required")
class TestBatchIntegration(unittest.TestCase):
    """Integration tests that make real OpenAI API calls"""
    
    def setUp(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.eval_instance = OpenAIBatchLLMAsJudgeEval(self.api_key, "gpt-4o-mini")  # Use cheaper model for testing
        self.test_prompt = "What are some effective coping strategies for anxiety?"
        self.test_base_response = "Some effective coping strategies include deep breathing, mindfulness meditation, and progressive muscle relaxation."
    
    def create_mock_responses(self):
        """Create mock InferenceSynapse responses for testing"""
        responses = []
        
        # Good response
        response1 = Mock(spec=InferenceSynapse)
        response1.output = "Deep breathing exercises, cognitive behavioral techniques, and regular exercise can help manage anxiety effectively."
        
        # Another good response
        response2 = Mock(spec=InferenceSynapse)
        response2.output = "Mindfulness practices, grounding techniques, and maintaining a support network are valuable anxiety management tools."
        
        # Poor/short response
        response3 = Mock(spec=InferenceSynapse)
        response3.output = "Just relax."
        
        # Empty response
        response4 = Mock(spec=InferenceSynapse)
        response4.output = ""
        
        responses.extend([response1, response2, response3, response4])
        return responses
    
    def test_create_batch_real_data(self):
        """Test create_batch with real therapeutic responses"""
        responses = self.create_mock_responses()
        miner_uids = [101, 102, 103, 104]
        request_id = f"integration_test_{int(time.time())}"
        
        batches = self.eval_instance.create_batch(
            self.test_prompt,
            self.test_base_response,
            request_id,
            responses,
            max_tokens_per_response=500,
            miner_uids=miner_uids
        )
        
        # Verify batch structure
        self.assertIsInstance(batches, list)
        self.assertGreater(len(batches), 0)
        
        batch, metadata = batches[0]
        self.assertIsInstance(batch, list)
        self.assertIsInstance(metadata, dict)
        
        # Verify request structure
        if batch:
            request = batch[0]
            self.assertIn("custom_id", request)
            self.assertIn("method", request)
            self.assertIn("url", request)
            self.assertIn("body", request)
            
            # Verify request content
            body = request["body"]
            self.assertEqual(body["model"], "gpt-4o-mini")
            self.assertIn("messages", body)
            self.assertEqual(len(body["messages"]), 2)  # system + user message
            
            # Verify judge prompt contains our test data
            user_message = body["messages"][1]["content"]
            self.assertIn(self.test_prompt, user_message)
            self.assertIn(self.test_base_response, user_message)
            self.assertIn("Deep breathing exercises", user_message)  # From response1
            self.assertIn("Mindfulness practices", user_message)  # From response2
    
    def test_queue_batch_real_api(self):
        """Test queue_batch with real OpenAI API calls"""
        # Create a small batch for testing
        responses = self.create_mock_responses()[:2]  # Only use first 2 responses
        miner_uids = [201, 202]
        request_id = f"queue_test_{int(time.time())}"
        
        # Create batch
        batches = self.eval_instance.create_batch(
            self.test_prompt,
            self.test_base_response,
            request_id,
            responses,
            max_tokens_per_response=200,
            miner_uids=miner_uids
        )
        
        self.assertGreater(len(batches), 0)
        batch, metadata = batches[0]
        
        # Queue the batch - this makes real API calls
        try:
            batch_response = self.eval_instance.queue_batch(batch, metadata)
            
            # Verify response structure
            self.assertIsNotNone(batch_response)
            self.assertIsNotNone(batch_response.id)
            self.assertIn("batch_", batch_response.id)
            
            # Check initial status
            self.assertIn(batch_response.status, ["validating", "in_progress", "finalizing", "completed"])
            
            print(f"✅ Successfully queued batch with ID: {batch_response.id}")
            print(f"   Status: {batch_response.status}")
            print(f"   Metadata: {metadata}")
            
            return batch_response.id
            
        except Exception as e:
            self.fail(f"Failed to queue batch: {str(e)}")
    
    def test_create_and_queue_integration(self):
        """Full integration test: create batch then queue it"""
        responses = self.create_mock_responses()[:3]  # Use 3 responses
        miner_uids = [301, 302, 303]
        request_id = f"full_test_{int(time.time())}"
        
        # Step 1: Create batch
        batches = self.eval_instance.create_batch(
            self.test_prompt,
            self.test_base_response,
            request_id,
            responses,
            max_tokens_per_response=300,
            miner_uids=miner_uids
        )
        
        self.assertIsInstance(batches, list)
        self.assertGreater(len(batches), 0)
        
        # Step 2: Queue each batch
        queued_batches = []
        for i, (batch, metadata) in enumerate(batches):
            try:
                batch_response = self.eval_instance.queue_batch(batch, metadata)
                queued_batches.append(batch_response)
                
                print(f"✅ Batch {i+1} queued successfully:")
                print(f"   ID: {batch_response.id}")
                print(f"   Status: {batch_response.status}")
                print(f"   Request count: {len(batch)}")
                
            except Exception as e:
                self.fail(f"Failed to queue batch {i+1}: {str(e)}")
        
        # Verify all batches were queued
        self.assertEqual(len(queued_batches), len(batches))
        for batch_response in queued_batches:
            self.assertIsNotNone(batch_response.id)
            self.assertIn("batch_", batch_response.id)
    
    def test_batch_with_large_responses(self):
        """Test batch creation with responses that exceed token limits"""
        # Create responses with varying lengths
        responses = []
        
        # Very long response that should be clipped
        long_response = Mock(spec=InferenceSynapse)
        long_response.output = "This is a very detailed therapeutic response. " * 200  # Very long
        
        # Normal response
        normal_response = Mock(spec=InferenceSynapse)
        normal_response.output = "A normal length therapeutic response about anxiety management."
        
        responses = [long_response, normal_response]
        miner_uids = [401, 402]
        request_id = f"large_test_{int(time.time())}"
        
        # Create batch with smaller token limit
        batches = self.eval_instance.create_batch(
            self.test_prompt,
            self.test_base_response,
            request_id,
            responses,
            max_tokens_per_response=100,  # Small limit to force clipping
            miner_uids=miner_uids
        )
        
        # Verify batch was created successfully
        self.assertGreater(len(batches), 0)
        batch, metadata = batches[0]
        
        # Queue the batch
        batch_response = self.eval_instance.queue_batch(batch, metadata)
        self.assertIsNotNone(batch_response)
        print(f"✅ Large response batch queued: {batch_response.id}")
    
    def test_error_handling_invalid_api_key(self):
        """Test error handling with invalid API key"""
        # Create instance with invalid key
        invalid_eval = OpenAIBatchLLMAsJudgeEval("invalid_key_123")
        
        responses = self.create_mock_responses()[:1]
        miner_uids = [501]
        request_id = f"error_test_{int(time.time())}"
        
        # Create batch should work (no API call)
        batches = invalid_eval.create_batch(
            self.test_prompt,
            self.test_base_response,
            request_id,
            responses,
            max_tokens_per_response=200,
            miner_uids=miner_uids
        )
        
        # Queue batch should fail with API error
        batch, metadata = batches[0]
        with self.assertRaises(Exception):
            invalid_eval.queue_batch(batch, metadata)


if __name__ == '__main__':
    # Print helpful information about running tests
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Integration tests will be skipped.")
        print("   To run integration tests:")
        print("   OPENAI_API_KEY=your_key python -m pytest tests/test_batch_integration.py -v")
    else:
        print("✅ OPENAI_API_KEY found. Running integration tests...")
        print("⚠️  These tests will make real API calls and may incur costs.")
    
    unittest.main()