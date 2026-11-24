"""Verification tests for the restricted onset/detection model, not a neural benchmark."""
import math
import random
import unittest
from experiments.verification.synthetic_likelihood import run as pilot


class LikelihoodTests(unittest.TestCase):
    def test_all_verification_checks(self):
        result=pilot.verify()
        self.assertEqual(result['cases'],189)
        self.assertLess(result['max_probability_error'],1e-12)
        self.assertLess(result['max_gradient_error'],1e-6)

    def test_zero_horizon_and_same_bin_detection(self):
        self.assertEqual(pilot.forward([],[]),([],1.0,0.0))
        det,u,v=pilot.forward([.2],[.3])
        self.assertAlmostEqual(det[0],.06)
        self.assertAlmostEqual(u,.8)
        self.assertAlmostEqual(v,.14)

    def test_undetected_does_not_mean_no_fraud(self):
        det,u,v=pilot.forward([.08]*20,[.3]*20)
        self.assertGreater(v,0)
        self.assertAlmostEqual(sum(det)+u+v,1)
        self.assertLess(sum(det)+u,1)
        posterior=pilot.responsibilities(.08,.3,20,False)
        self.assertAlmostEqual(posterior[0],u/(u+v))
        self.assertAlmostEqual(sum(posterior[1:]),v/(u+v))

    def test_observation_indicator_is_separate_from_onset_and_boundary(self):
        rows=[(1,3,3,1),(2,6,3,0),(5,7,3,0)]
        self.assertEqual(pilot.observed_counts(rows,3),[0,0,1,2])

    def test_rate_swap_changes_latent_risk_but_not_detection_likelihood(self):
        counts=[3,4,2,1,5]
        self.assertAlmostEqual(pilot.nll(counts,.08,.3),pilot.nll(counts,.3,.08),places=12)
        self.assertGreater(pilot.risk(.3,.08,20)-pilot.risk(.08,.3,20),.7)

    def test_log_likelihood_handles_ordinary_probability_underflow(self):
        log_det,log_u,log_v=pilot.log_forward([.8]*1000,[.9]*1000)
        self.assertTrue(math.isfinite(log_det[-1]))
        self.assertTrue(math.isfinite(pilot.logadd(log_u,log_v)))
        self.assertEqual(pilot.forward([.8]*1000,[.9]*1000)[0][-1],0)

    def test_analytic_gradients_for_multiple_censoring_patterns(self):
        rng=random.Random(44)
        for _ in range(15):
            counts=[rng.randint(0,20) for _ in range(8)]
            a,b=rng.uniform(.05,.7),rng.uniform(.05,.7)
            h=1e-6
            actual=pilot.nll_gradient(counts,a,b)
            expected=((pilot.nll(counts,a+h,b)-pilot.nll(counts,a-h,b))/(2*h),
                      (pilot.nll(counts,a,b+h)-pilot.nll(counts,a,b-h))/(2*h))
            for x,y in zip(actual,expected):
                self.assertAlmostEqual(x,y,places=6)

    def test_known_delay_population_objective_recovers_onset_parameter(self):
        det,u,v=pilot.forward([.08]*20,[.3]*20)
        expected=det+[u+v]
        fitted=pilot.bounded_minimum(lambda a:pilot.nll(expected,a,.3),.04,.15)
        self.assertAlmostEqual(fitted,.08,places=6)


if __name__=='__main__':
    unittest.main()
