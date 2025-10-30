import unittest
from unittest.mock import Mock, patch, mock_open
from BetterTherapy.protocol import InferenceSynapse
from evals.batch import OpenAIBatchLLMAsJudgeEval


class TestOpenAIBatchLLMAsJudgeEval(unittest.TestCase):
    
    @patch('evals.batch.OpenAI')
    def setUp(self, mock_openai):
        self.api_key = "test-api-key"
        self.mock_client = Mock()
        mock_openai.return_value = self.mock_client
        self.eval_instance = OpenAIBatchLLMAsJudgeEval(self.api_key)
    
    @patch('evals.batch.OpenAI')
    def test_init(self, mock_openai):
        eval_instance = OpenAIBatchLLMAsJudgeEval("test-key")
        self.assertEqual(eval_instance.judge_model, "gpt-4o")
        self.assertIsNone(eval_instance.base_response)
        
    @patch('evals.batch.OpenAI')
    def test_init_with_custom_model(self, mock_openai):
        custom_eval = OpenAIBatchLLMAsJudgeEval("test-key", "gpt-3.5-turbo")
        self.assertEqual(custom_eval.judge_model, "gpt-3.5-turbo")
    
    def test_create_judge_prompt_basic(self):
        prompt = "What is therapy?"
        base_response = "Therapy is a treatment method."
        responses = ["Response 1", "Response 2", None, ""]
        
        result = self.eval_instance.create_judge_prompt(prompt, base_response, responses)
        
        self.assertIsInstance(result, str)
        self.assertIn(prompt, result)
        self.assertIn(base_response, result)
        self.assertIn("Therapist 1: Response 1", result)
        self.assertIn("Therapist 2: Response 2", result)
        self.assertIn("Therapist 3:", result)
        self.assertIn("Therapist 4:", result)
        self.assertIn("JSON", result)
        self.assertIn("scores", result)
    
    def test_create_judge_prompt_empty_responses(self):
        prompt = "Test prompt"
        base_response = "Base response"
        responses = []
        
        result = self.eval_instance.create_judge_prompt(prompt, base_response, responses)
        
        self.assertIsInstance(result, str)
        self.assertIn(prompt, result)
        self.assertIn(base_response, result)
    
    def test_create_judge_prompt_none_responses(self):
        prompt = "Test prompt"
        base_response = "Base response"
        responses = [None, None, None]
        
        result = self.eval_instance.create_judge_prompt(prompt, base_response, responses)
        
        self.assertIsInstance(result, str)
        self.assertIn("Therapist 1:", result)
        self.assertIn("Therapist 2:", result)
        self.assertIn("Therapist 3:", result)
    
    @patch('evals.batch.count_and_clip_tokens')
    def test_create_batch_basic(self, mock_count_tokens):
        mock_count_tokens.return_value = (100, "clipped response")
        
        prompt = "Test prompt"
        base_response = "Base response"
        request_id = "test-123"
        
        response1 = Mock(spec=InferenceSynapse)
        response1.output = "Response 1"
        response2 = Mock(spec=InferenceSynapse)
        response2.output = "Response 2"
        
        responses = [response1, response2]
        miner_uids = [1, 2]
        max_tokens_per_response = 500
        
        result = self.eval_instance.create_batch(
            prompt, base_response, request_id, responses, 
            max_tokens_per_response, miner_uids
        )
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        batch, metadata = result[0]
        self.assertIsInstance(batch, list)
        self.assertIsInstance(metadata, dict)
        
        if batch:
            request = batch[0]
            self.assertIn("custom_id", request)
            self.assertIn("method", request)
            self.assertIn("url", request)
            self.assertIn("body", request)
            self.assertEqual(request["method"], "POST")
            self.assertEqual(request["url"], "/v1/chat/completions")
    
    @patch('evals.batch.count_and_clip_tokens')
    def test_create_batch_empty_responses(self, mock_count_tokens):
        mock_count_tokens.return_value = (0, "")
        
        response1 = Mock(spec=InferenceSynapse)
        response1.output = None
        response2 = Mock(spec=InferenceSynapse)
        response2.output = ""
        
        responses = [response1, response2]
        miner_uids = [1, 2]
        
        result = self.eval_instance.create_batch(
            "prompt", "base", "id", responses, 500, miner_uids
        )
        
        self.assertIsInstance(result, list)
    
    @patch('evals.batch.count_and_clip_tokens')
    def test_create_batch_token_limit(self, mock_count_tokens):
        mock_count_tokens.return_value = (7000, "very long response")
        
        response = Mock(spec=InferenceSynapse)
        response.output = "Long response"
        
        responses = [response]
        miner_uids = [1]
        
        result = self.eval_instance.create_batch(
            "prompt", "base", "id", responses, 500, miner_uids
        )
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
    
    @patch('evals.batch.count_and_clip_tokens')
    def test_create_batch_max_request_limit(self, mock_count_tokens):
        mock_count_tokens.return_value = (100, "response")
        
        responses = []
        miner_uids = []
        for i in range(15):
            response = Mock(spec=InferenceSynapse)
            response.output = f"Response {i}"
            responses.append(response)
            miner_uids.append(i)
        
        result = self.eval_instance.create_batch(
            "prompt", "base", "id", responses, 500, miner_uids, max_request_per_batch=10
        )
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
    
    @patch('builtins.open', new_callable=mock_open)
    def test_queue_batch(self, mock_file_open):
        mock_file_obj = Mock()
        mock_file_obj.id = "file-123"
        self.eval_instance.judge_client.files.create.return_value = mock_file_obj
        
        mock_batch_response = Mock()
        mock_batch_response.id = "batch-456"
        self.eval_instance.judge_client.batches.create.return_value = mock_batch_response
        
        batch = [{"test": "data"}]
        metadata = {"key": "value"}
        
        result = self.eval_instance.queue_batch(batch, metadata)
        
        self.assertEqual(result, mock_batch_response)
        self.eval_instance.judge_client.files.create.assert_called_once()
        self.eval_instance.judge_client.batches.create.assert_called_once()
        
        call_args = self.eval_instance.judge_client.batches.create.call_args[1]
        self.assertEqual(call_args["input_file_id"], "file-123")
        self.assertEqual(call_args["endpoint"], "/v1/chat/completions")
        self.assertEqual(call_args["completion_window"], "24h")
        self.assertEqual(call_args["metadata"], metadata)
    
    def test_query_batch_completed(self):
        mock_batch = Mock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "output-file-123"
        self.eval_instance.judge_client.batches.retrieve.return_value = mock_batch
        
        mock_file_response = Mock()
        mock_file_response.text = "line1\nline2\nline3"
        self.eval_instance.judge_client.files.content.return_value = mock_file_response
        
        responses, batch = self.eval_instance.query_batch("batch-123")
        
        self.assertEqual(responses, ["line1", "line2", "line3"])
        self.assertEqual(batch, mock_batch)
        self.eval_instance.judge_client.batches.retrieve.assert_called_once_with("batch-123")
        self.eval_instance.judge_client.files.content.assert_called_once_with("output-file-123")
    
    def test_query_batch_not_completed(self):
        mock_batch = Mock()
        mock_batch.status = "processing"
        mock_batch.errors = None
        self.eval_instance.judge_client.batches.retrieve.return_value = mock_batch
        
        responses, batch = self.eval_instance.query_batch("batch-123")
        
        self.assertIsNone(responses)
        self.assertIsNone(batch)
        self.eval_instance.judge_client.batches.retrieve.assert_called_once_with("batch-123")
    
    def test_query_batch_with_errors(self):
        mock_batch = Mock()
        mock_batch.status = "failed"
        mock_error = Mock()
        mock_error.to_json.return_value = '{"error": "test error"}'
        mock_batch.errors = mock_error
        self.eval_instance.judge_client.batches.retrieve.return_value = mock_batch
        
        responses, batch = self.eval_instance.query_batch("batch-123")
        
        self.assertIsNone(responses)
        self.assertIsNone(batch)
    
    def test_query_batch_completed_no_output_file(self):
        mock_batch = Mock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = None
        self.eval_instance.judge_client.batches.retrieve.return_value = mock_batch
        
        responses, batch = self.eval_instance.query_batch("batch-123")
        
        self.assertIsNone(responses)
        self.assertIsNone(batch)


if __name__ == '__main__':
    unittest.main()