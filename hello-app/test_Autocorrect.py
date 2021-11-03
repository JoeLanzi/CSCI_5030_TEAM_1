import unittest

class TestAutocorrect(unittest.TestCase):

    def test_language_detect(self):
        pass

    def test_load_dictionary(self):
        pass

    def test_correct(self):
        checker = Autocorrect()
        with open('test_words.txt','r') as tw:
            test_words = {}
            for line in tw.readlines():
                line_list = line.strip('\n').split('\t')
                test_words[line_list[0]] = line_list[1]
        for each in test_words:
            self.assertEqual(checker.correct(each), test_words[each])

    def test_suggestion(self):
        pass

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)