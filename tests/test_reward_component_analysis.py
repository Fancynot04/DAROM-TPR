import tempfile
import unittest
from pathlib import Path

from scripts.analyze_reward_components import analyse, component_stats, jsonl_records


def term(name, raw, weighted, enabled=True, active=True, clipped=False, clipped_value=None):
    return {'name': name, 'raw': raw, 'weighted': weighted, 'enabled': enabled,
            'active': active, 'clipped': clipped, 'clipped_value': clipped_value}


class RewardComponentAnalysisTests(unittest.TestCase):
    def test_b0_b1_b2_b3_and_episode_discount_reset(self):
        records=[]
        profiles={'b0':[term('success',0,0),term('collision',-100,-100,active=True)],
                  'b1':[term('efficiency',-2,-.005,clipped=True,clipped_value=-1)],
                  'b2':[term('route',-.2,-.001)],
                  'b3':[term('efficiency',-2,-.005,clipped=True,clipped_value=-1),term('route',-.2,-.001)]}
        for group, terms in profiles.items():
            records.append({'group':group,'episode_id':'a','step':0,'gamma':.9,'reward':sum(x['weighted'] for x in terms),'outcome':None,'terms':terms})
        episodes,samples,issues=analyse(records)
        self.assertFalse(issues)
        self.assertEqual(len(episodes),4)
        self.assertAlmostEqual(episodes.loc[episodes.group=='b3','total_reward'].iloc[0],-.006)
        stats=component_stats(samples, episodes)
        self.assertEqual(stats.loc[stats.name=='efficiency','clipped_samples'].sum(),2)

    def test_zero_and_cancelling_terms_are_consistent(self):
        records=[{'group':'x','episode_id':'one','step':0,'gamma':.5,'reward':0,'outcome':None,
                  'terms':[term('success',1,1),term('route',-1,-1)]},
                 {'group':'x','episode_id':'two','step':0,'gamma':.5,'reward':2,'outcome':None,
                  'terms':[term('success',2,2)]}]
        episodes,_,issues=analyse(records)
        self.assertFalse(issues)
        self.assertTrue(episodes.loc[episodes.episode_id=='one','success_discounted_abs_share'].iloc[0] == .5)

    def test_missing_or_mismatched_payment_is_reported(self):
        records=[{'group':'x','episode_id':'one','step':0,'gamma':.9,'reward':1,'outcome':None,
                  'terms':[term('route',0,0,enabled=False,active=False)]}]
        _,_,issues=analyse(records)
        self.assertTrue(any('term sum' in issue for issue in issues))

    def test_jsonl_without_step_terms_warns_instead_of_creating_zero_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episodes.jsonl'
            path.write_text('{"reward": 3.0}\n')
            records, messages = jsonl_records(path, .99)
        self.assertEqual(records, [])
        self.assertTrue(any('no per-step reward_terms' in message for message in messages))


if __name__=='__main__': unittest.main()
