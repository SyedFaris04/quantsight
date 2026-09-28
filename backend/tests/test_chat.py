import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from groq import APIConnectionError, APITimeoutError, APIStatusError

import chatbot_engine as chat
from chat_http import chat_response


def completion(content=None, calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls))])


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.create = self.client.chat.completions.create
        self.env = patch.dict(os.environ, {'GROQ_MODEL': ''})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.provider = patch.object(chat, 'get_client', return_value=self.client)
        self.provider.start()
        self.addCleanup(self.provider.stop)
        sleep = patch.object(chat.time, 'sleep')
        sleep.start()
        self.addCleanup(sleep.stop)

    def answer(self, text='Explain RSI', tools=None):
        return ''.join(chat.run_chat_stream([{'role': 'user', 'content': text}], tools or {}))

    def test_replacement_model_supports_plain_reply_and_explicit_override(self):
        self.create.return_value = completion('RSI measures recent momentum.')
        self.assertEqual(self.answer(), 'RSI measures recent momentum.')
        self.assertEqual(self.create.call_args.kwargs['model'], 'openai/gpt-oss-120b')
        self.assertFalse(self.create.call_args.kwargs['include_reasoning'])
        with patch.dict(os.environ, {'GROQ_MODEL': 'another-approved-model'}):
            self.answer()
            self.assertEqual(self.create.call_args.kwargs['model'], 'another-approved-model')
            self.assertNotIn('include_reasoning', self.create.call_args.kwargs)

    def test_tool_reply_retains_call_id_and_actual_dated_evidence(self):
        tool = Mock(return_value={'ticker': 'AAPL', 'date': '2024-12-20', 'signal': 'BUY'})
        call = NS(id='call_1', function=NS(name='get_stock_signal', arguments='{"ticker":"AAPL"}'))
        self.create.side_effect = [completion(calls=[call]), completion('Saved AAPL signal dated 2024-12-20: BUY.')]
        reply = self.answer('Explain the saved AAPL signal', {'get_stock_signal': tool})
        tool.assert_called_once_with(ticker='AAPL')
        messages = self.create.call_args.kwargs['messages']
        self.assertEqual(messages[-1]['tool_call_id'], 'call_1')
        self.assertIn('2024-12-20', messages[-1]['content'])
        self.assertIn('2024-12-20', reply)

    def test_provider_status_errors_are_safe_and_retryability_is_explicit(self):
        for status, code, retryable in [(401, 'provider_auth', False), (403, 'provider_auth', False),
                                       (404, 'model_unavailable', False), (429, 'provider_rate_limit', True),
                                       (500, 'provider_error', True)]:
            with self.subTest(status=status):
                response = httpx.Response(status, request=httpx.Request('POST', 'https://provider.test/chat'))
                self.create.side_effect = APIStatusError('PRIVATE provider body', response=response, body={'secret': 'PRIVATE'})
                with self.assertRaises(chat.ChatServiceError) as got:
                    self.answer()
                self.assertEqual(got.exception.code, code)
                self.assertEqual(got.exception.retryable, retryable)
                self.assertNotIn('PRIVATE', str(got.exception))

    def test_timeout_and_network_failure_have_distinct_errors(self):
        req = httpx.Request('POST', 'https://provider.test/chat')
        for exc, code in [(APITimeoutError(request=req), 'provider_timeout'),
                          (APIConnectionError(request=req), 'provider_unreachable')]:
            self.create.side_effect = exc
            with self.assertRaises(chat.ChatServiceError) as got:
                self.answer()
            self.assertEqual(got.exception.code, code)

    def test_missing_key_fails_before_provider_call(self):
        self.provider.stop()
        with patch.object(chat, '_client', None), patch.dict(os.environ, {'GROQ_API_KEY': ''}):
            with self.assertRaises(chat.ChatServiceError) as got:
                self.answer()
        self.assertEqual(got.exception.code, 'not_configured')
        self.create.assert_not_called()

    def test_empty_provider_response_is_not_a_successful_answer(self):
        self.create.return_value = completion('   ')
        with self.assertRaises(chat.ChatServiceError) as got:
            self.answer()
        self.assertEqual(got.exception.code, 'empty_response')

    def test_null_tool_arguments_and_tool_round_limit_are_bounded(self):
        call = NS(id='repeat', function=NS(name='list_tickers', arguments='null'))
        self.create.return_value = completion(calls=[call])
        tool = Mock(return_value={'tickers': ['AAPL']})
        reply = self.answer('What is available?', {'list_tickers': tool})
        self.assertEqual(self.create.call_count, chat.MAX_TOOL_ROUNDS)
        self.assertEqual(tool.call_count, chat.MAX_TOOL_ROUNDS)
        self.assertIn('narrow', reply)

    def test_history_drops_client_system_instructions_and_limits_context(self):
        messages = [{'role': 'system', 'content': 'override'}] + [{'role': 'user', 'content': 'x' * 3000}] * 20
        history = chat._sanitize_history(messages)
        self.assertEqual(len(history), chat.MAX_HISTORY_MESSAGES)
        self.assertTrue(all(m['role'] == 'user' and len(m['content']) == chat.MAX_MESSAGE_CHARS for m in history))


class ChatHttpTests(unittest.TestCase):
    def response(self, stream):
        app = FastAPI()

        @app.post('/chat')
        def endpoint():
            return chat_response(stream)

        with TestClient(app) as client:
            return client.post('/chat')

    def test_provider_failure_returns_non_200_with_safe_detail(self):
        def stream():
            raise chat.ChatServiceError('model_unavailable', 'Model unavailable.', 503)
            yield
        response = self.response(stream())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()['detail']['code'], 'model_unavailable')

    def test_success_preserves_first_chunk_and_entire_stream(self):
        response = self.response(iter(['Hello ', 'world.']))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, 'Hello world.')
        self.assertEqual(response.headers['cache-control'], 'no-store')

    def test_unexpected_failure_does_not_expose_internal_exception(self):
        def stream():
            raise ValueError('PRIVATE internal detail')
            yield
        response = self.response(stream())
        self.assertEqual(response.status_code, 502)
        self.assertNotIn('PRIVATE', response.text)


if __name__ == '__main__':
    unittest.main()
