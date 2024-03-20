from typing import Optional
from collections import deque
from queue import Queue
import copy


class History:

    def __init__(self, tokenizer, history):
        '''
        init from a list of dict
        '''
        # use deque to meet some special situation
        self.input_history = deque()
        self.tokenizer = tokenizer
        if history:
            self._transfer_from_list(history)

    def _transfer_from_list(self, history):
        for message in history:
            content = message.get("content")
            # the token result may not be equal to the result model gen
            message.update(self.tokenizer(content))
            self.input_history.append(message)

    def append(self, message):
        content = message.get("content")
        if "input_ids" not in message or "attention_mask" not in message:
            message.update(self.tokenizer(content))
        self.input_history.append(message)

    def append_left(self, message):
        content = message.get("content")
        if "input_ids" not in message or "attention_mask" not in message:
            message.update(self.tokenizer(content))
        self.input_history.appendleft(message)

    def pop(self):
        x = self.input_history.pop()
        return x

    def pop_left(self):
        x = self.pop_left()
        return x

    def update(self, message):
        self.input_history.pop()
        self.append(message)

    def __len__(self):
        return self.input_history.__len__()

    def __str__(self):
        return self.input_history.__str__()

    def __copy__(self):
        new_instance = type(self)(self.tokenizer, [])
        new_instance.input_history = copy.copy(self.input_history)
        return new_instance

    def __deepcopy__(self, memodict={}):
        new_instance = type(self)(self.tokenizer, [])
        new_instance.input_history = copy.deepcopy(self.input_history)
        return new_instance


class TelechatIterTextStreamer:
    """
    With reference to the TextIterStreamers in transformers, we have rewritten this class
    """

    def __init__(
            self, tokenizer, history: History = None, skip_prompt: bool = False, timeout: Optional[float] = None,
            **decode_kwargs
    ):

        self.tokenizer = tokenizer
        self.history = history
        self.skip_prompt = skip_prompt
        self.timeout = timeout
        self.decode_kwargs = decode_kwargs

        self.text_queue = Queue()
        self.cache_time = 0
        self.text_until = ""
        self.token_until = []
        self.stop_signal = None
        self.next_tokens_are_prompt = True

        self.history.append({"role": "bot", "content": self.text_until})

    def put(self, value):
        """
        put printable text into queue
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        if value[-1] == self.tokenizer.eos_token_id:
            return

        # there may be some smart way to decode.
        self.token_until.extend(value.tolist())
        text = self.tokenizer.decode(self.token_until, **self.decode_kwargs)


        if self._is_printable(text) or self.cache_time >= 6:
            output_text = text[len(self.text_until):]
            self.text_until = text

        else:
            self.cache_time+=1
            return

        self.on_finalized_text(output_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        text = self.tokenizer.decode(self.token_until, **self.decode_kwargs)
        output_text = text[len(self.text_until):]
        self.text_until = text
        self.on_finalized_text(output_text, stream_end=True)
        self.clear_cache()

    def clear_cache(self):
        self.cache_time = 0
        self.token_until = []
        self.text_until = ""
        self.history = None
        self.next_tokens_are_prompt = True

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the text tuple in the queue."""
        self.history.update({"role": "bot", "content": self.text_until, "input_ids": self.token_until,
                             "attention_mask": [1] * len(self.token_until)})
        self.text_queue.put((text, self.history), timeout=self.timeout)
        if stream_end:
            self.text_queue.put((self.stop_signal, self.history), timeout=self.timeout)

    @staticmethod
    def _is_printable(cp):
        """Checks whether tokens can be decoded or not"""
        if "�" in cp:
            return False
        return True

    def __iter__(self):
        return self

    def __next__(self):
        value_now, history_until = self.text_queue.get(timeout=self.timeout)
        if value_now == self.stop_signal:
            raise StopIteration()
        else:
            return value_now, history_until
