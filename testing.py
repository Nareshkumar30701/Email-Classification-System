import unittest
from src import trans_to_en, clean_text, vectorize_text, prepare_data, train_and_evaluate
import pandas as pd

class TestSrc(unittest.TestCase):
    def setUp(self):
        # Setup data for testing
        self.sample_data = pd.DataFrame({
            'Interaction content': ['Thank you for your email', 'Hello, how can I help you?', 'This is a test message'],
            'Ticket Summary': ['RE: Issue with the app', 'Hello there', 'Test case summary'],
            'Type 2': ['issue', 'greeting', 'test']
        })
        self.sample_data['Interaction content'] = self.sample_data['Interaction content'].astype('U')
        self.sample_data['Ticket Summary'] = self.sample_data['Ticket Summary'].astype('U')
        self.sample_data['y'] = self.sample_data['Type 2']

    def test_trans_to_en(self):
        # Testing translation component
        translated_texts = trans_to_en(self.sample_data['Ticket Summary'].tolist())
        for text in translated_texts:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)

    def test_clean_text(self):
        # Testing noise removal component
        cleaned_data = clean_text(self.sample_data.copy())
        for col in ['ts', 'ic']:
            self.assertIn(col, cleaned_data.columns)
            for text in cleaned_data[col]:
                self.assertIsInstance(text, str)

    def test_model_building(self):
        # Testing model building components
        cleaned_data = clean_text(self.sample_data.copy())
        X = vectorize_text(cleaned_data)
        X_train, X_test, y_train, y_test = prepare_data(cleaned_data, X)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertGreater(len(y_train), 0)
        self.assertGreater(len(y_test), 0)
        train_and_evaluate(X_train, X_test, y_train, y_test)
        
if __name__ == '__main__':
    unittest.main()
