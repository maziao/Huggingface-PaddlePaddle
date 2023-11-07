from queue import Queue
from typing import TYPE_CHECKING, Optional


class BaseStreamer:
    """
    Base class from which `.generate()` streamers should inherit.
    """

    def put(self, value):
        """Function that is called by `.generate()` to push new tokens"""
        raise NotImplementedError()

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        raise NotImplementedError()


class TextStreamer(BaseStreamer):
    """
    Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

        >>> tok = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextStreamer(tok)

        >>> # Despite returning the usual output, the streamer will also print the generated text to stdout.
        >>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
        An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
        ```
    """

    def __init__(self, tokenizer: 'AutoTokenizer', skip_prompt: bool=False,
        **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        """
        Recives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError('TextStreamer only supports batch size 1')
        elif len(value.shape) > 1:
            value = value[0]
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        if text.endswith('\n'):
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = text[self.print_len:text.rfind(' ') + 1]
            self.print_len += len(printable_text)
        self.on_finalized_text(printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs
                )
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ''
        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool=False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        print(text, flush=True, end='' if not stream_end else None)


class TextIteratorStreamer(TextStreamer):
    """
    Streamer that stores print-ready text in a queue, to be used by a downstream application as an iterator. This is
    useful for applications that benefit from acessing the generated text in a non-blocking way (e.g. in an interactive
    Gradio demo).

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        timeout (`float`, *optional*):
            The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
            in `.generate()`, when it is called in a separate thread.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        >>> from threading import Thread

        >>> tok = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextIteratorStreamer(tok)

        >>> # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        >>> generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        >>> thread = Thread(target=model.generate, kwargs=generation_kwargs)
        >>> thread.start()
        >>> generated_text = ""
        >>> for new_text in streamer:
        ...     generated_text += new_text
        >>> generated_text
        'An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,'
        ```
    """

    def __init__(self, tokenizer: 'AutoTokenizer', skip_prompt: bool=False,
        timeout: Optional[float]=None, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool=False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value
