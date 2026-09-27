import unittest
import numpy as np
import pandas as pd

from research.finbert_pilot import ordered_probabilities, align_parent, aggregate_scores


class FinbertResearchTests(unittest.TestCase):
    def test_label_order_comes_from_model_config(self):
        actual = ordered_probabilities([[.2, .1, .7]], {0: 'neutral', 1: 'negative', 2: 'positive'})
        np.testing.assert_allclose(actual, [[.7, .1, .2]])

    def test_invalid_model_labels_and_probabilities_rejected(self):
        for probabilities, labels in [([[.2, .3, .5]], {0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'}),
                                      ([[.5, .5, .5]], dict(enumerate(['positive', 'negative', 'neutral']))),
                                      ([[np.nan, .2, .8]], dict(enumerate(['positive', 'negative', 'neutral']))),
                                      ([[-.1, .5, .6]], dict(enumerate(['positive', 'negative', 'neutral'])))]:
            with self.assertRaises(ValueError):
                ordered_probabilities(probabilities, labels)

    def evaluation(self):
        return pd.DataFrame({'ticker': ['A', 'B'], 'date': ['2023-01-03'] * 2,
                             'signal': [1, 0], 'news_observed': [1., 0.]})

    def test_parent_predictions_align_by_keys_not_row_position(self):
        expected = self.evaluation()
        parent = expected.assign(prediction=[.8, .2]).iloc[::-1]
        actual = align_parent(expected, parent)
        self.assertEqual(actual.prediction.tolist(), [.8, .2])

    def test_parent_mismatches_cannot_silently_change_sample(self):
        expected = self.evaluation()
        wrong_label = expected.copy(); wrong_label.loc[0, 'signal'] = 0
        wrong_coverage = expected.copy(); wrong_coverage.loc[0, 'news_observed'] = 0
        wrong_key = expected.copy(); wrong_key.loc[0, 'ticker'] = 'C'
        duplicates = pd.concat([expected.iloc[:1], expected.iloc[:1]])
        for parent in [wrong_label, wrong_coverage, wrong_key, duplicates, expected.iloc[:1]]:
            with self.assertRaises(ValueError):
                align_parent(expected, parent)

    def news(self):
        return pd.DataFrame({'ticker': ['A', 'A', 'B'], 'date': ['2023-01-03'] * 3,
                             'article_id': ['1', '2', '1'], 'headline': ['gain', 'loss', 'gain']})

    def scores(self):
        return pd.DataFrame({'headline': ['gain', 'loss'], 'positive': [.8, .1],
                             'negative': [.1, .7], 'neutral': [.1, .2]})

    def test_aggregation_preserves_ticker_links_and_coverage(self):
        actual = aggregate_scores(self.news(), self.scores()).set_index('ticker')
        self.assertAlmostEqual(actual.loc['A', 'news_polarity_mean'], .05)
        self.assertAlmostEqual(actual.loc['A', 'news_positive_share'], .5)
        self.assertAlmostEqual(actual.loc['A', 'news_negative_share'], .5)
        self.assertAlmostEqual(actual.loc['A', 'news_log_count'], np.log1p(2))
        self.assertAlmostEqual(actual.loc['B', 'news_polarity_mean'], .7)
        self.assertEqual(actual.loc['B', 'news_polarity_std'], 0)
        self.assertTrue(actual.news_observed.eq(1).all())

    def test_missing_or_duplicate_scores_fail_instead_of_becoming_neutral(self):
        for scores in [self.scores().iloc[:1], pd.concat([self.scores(), self.scores().iloc[:1]])]:
            with self.assertRaises(ValueError):
                aggregate_scores(self.news(), scores)


if __name__ == '__main__':
    unittest.main()
