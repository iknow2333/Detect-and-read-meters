# 这段代码定义了一个类 StringLabelConverter，用于在字符串和标签之间进行转换，主要应用于字符识别任务中，如光学字符识别（OCR）。这个转换器特别考虑了CTC（Connectionist Temporal Classification）的需求，这是一种常用于序列识别问题的训练技术，特别是在没有分割字符的情况下进行文本识别。

# 以下是该类的详细解释：
# 初始化（init）:

#     alphabet: 包含可能字符的字符串。
#     ignore_case: 布尔值，指示是否忽略大小写。如果为True，则将alphabet转换为小写。
#     该方法还为CTC增加一个'-'字符作为“blank”符号，并创建一个字典self.dict，将每个字符映射到一个唯一的索引（索引从1开始，因为索引0保留给CTC的“空白”字符）。

# 编码（encode）:

#     将文本字符串转换为长整型张量（torch.LongTensor），它们可以直接用于模型训练。
#     这个函数支持批量处理，可以同时转换多个字符串。
#     对于每个字符，如果它是阿拉伯文（由unicodedata的name方法检查），字符串将被反转，因为阿拉伯文是从右到左书写的。
#     返回编码后的张量和每个文本的实际长度。

# 解码（decode）:

#     将编码的长整型张量转换回字符串。
#     这个函数同样支持批量处理，可以同时解码多个文本。
#     如果raw参数为False，则解码时会去除重复的字符和空白字符，这对于CTC的解码是必要的，因为CTC可能会输出重复的字符。
#     如果文本中的第一个字符是阿拉伯文，解码后的字符串将被反转回来，以保持正确的阅读顺序。

# 此类是在处理字符识别任务，尤其是在使用CTC损失函数时，对文本数据进行预处理和后处理的一个有用工具。通过处理包括阿拉伯文在内的不同语言和字符集，这个转换器能够支持多语言OCR系统的开发。

import torch
import unicodedata as ud

# keys = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@!"#$%&\'[]()+,-./:;=?´ÉÈ'
with open('util/codec.txt', 'r') as f:
    keys = f.readlines()[0]


class StringLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        length = []
        result = []
        for item in text:
            if len(item) > 0 and 'ARABIC' in ud.name(item[0]):
                item = item[::-1]
            length.append(len(item))
            r = []
            for char in item:
                index = self.dict[char]
                r.append(index)
            result.append(r)

        max_len = 0
        for r in result:
            if len(r) > max_len:
                max_len = len(r)

        result_temp = []
        for r in result:
            for i in range(max_len - len(r)):
                r.append(0)
            result_temp.append(r)

        text = result_temp
        return torch.LongTensor(text), torch.LongTensor(length)

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                output = ''.join([self.alphabet[i - 1] for i in t])
                if len(output) > 0 and 'ARABIC' in ud.name(output[0]):
                    output = output[::-1]
                return output
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])

                output = ''.join(char_list)
                if len(output) > 0 and 'ARABIC' in ud.name(output[0]):
                    output = output[::-1]
                return output
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.LongTensor([l]), raw=raw))
                index += l
            return texts
