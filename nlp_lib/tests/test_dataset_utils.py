"""
Run the tests with:

python -m unittest discover -s nlp_lib/tests -p "test_*.py"
"""

import unittest
import torch
from nlp_lib.dataset_utils import tokenize_sentence_pair_sts_dataset, cosine_sim, collate_fn
from transformers import AutoTokenizer


def hello_world_from_nlp_lib():
    print("Hello world from nlp_lib!")


class TestBertUtilities(unittest.TestCase):

    def test_tokenize_sentence_pair_sts_dataset(self):
        # Dummy dataset and tokenizer
        dataset = [("Hello world", "How are you?", 1.0), ("It's sunny", "Nice weather", 2.5)]
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        tokenized_dataset = tokenize_sentence_pair_sts_dataset(dataset, tokenizer)

        # Assert that tokenized_dataset has the expected length
        self.assertEqual(len(tokenized_dataset), len(dataset))

        # Assert that each item in tokenized_dataset has 'input_ids', 'attention_mask', etc.
        for item in tokenized_dataset:
            self.assertIn('input_ids', item)
            self.assertIn('attention_mask', item)
            self.assertIn('token_type_ids', item)
            self.assertIn('score', item)

    def test_cosine_sim(self):
        # Dummy data
        a = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = cosine_sim(a, b)

        # Assert that result is a 2x2 tensor
        self.assertEqual(result.shape, (2, 2))

        print(result)

        # Assert that the diagonal (i.e., cosine similarity with itself) is approximately 1
        for i in range(result.shape[0]):
            self.assertAlmostEqual(result[i][i].item(), 1.0, delta=1e-2)

    def test_cosine_sim_2(self):
      a = torch.tensor([[1, 0], [0, 1], [1, 1]]).float()
      b = torch.tensor([[1, 0], [0, 1], [-1, -1]]).float()
      expected_result = torch.tensor([[1, 0, -0.7071], [0, 1, -0.7071], [0.7071, 0.7071, -1]])

      result = cosine_sim(a, b)
      torch.testing.assert_close(result, expected_result, atol=1e-4, rtol=1e-4)



    def test_collate_fn(self):
        # Dummy batch
        batch = [{
            'input_ids': torch.tensor([1, 2, 3]),
            'attention_mask': torch.tensor([1, 1, 0]),
            'token_type_ids': torch.tensor([0, 0, 1]),
            'score': torch.tensor(2.5)
        }, {
            'input_ids': torch.tensor([4, 5, 6]),
            'attention_mask': torch.tensor([1, 1, 1]),
            'token_type_ids': torch.tensor([0, 1, 1]),
            'score': torch.tensor(3.5)
        }]
        collated_batch = collate_fn(batch)

        # Assert that 'input_ids', 'attention_mask', etc. in collated_batch have expected shapes
        self.assertEqual(collated_batch['input_ids'].shape, (2, 3))
        self.assertEqual(collated_batch['attention_mask'].shape, (2, 3))
        self.assertEqual(collated_batch['token_type_ids'].shape, (2, 3))
        self.assertEqual(collated_batch['score'].shape, (2,))

if __name__ == '__main__':
    unittest.main()

# # To run the tests in the notebook environment, uncomment the following lines:
# suite = unittest.TestLoader().loadTestsFromTestCase(TestBertUtilities)
# unittest.TextTestRunner(verbosity=2).run(suite)



