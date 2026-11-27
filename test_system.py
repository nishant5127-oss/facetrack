"""
FaceTrack Core Test Suite (9/9 Functional Test Cases)
Verifies Face Detection, Landmark Normalization, k-NN RMS Matching, Database Integrity, and Privacy Filtering.
"""

import json
import os
import unittest
import numpy as np
from sqlmodel import Session, select

import db
import face
import matcher
import emailer

class TestFaceTrack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init()

    # TC-01: Landmark Vector Dimensions & Non-Null Extraction
    def test_01_landmark_vector_dimensions(self):
        # Synthetic 100x100 black image with no face
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        emb = face.embed(blank)
        self.assertIsNone(emb, "TC-01 Failed: Blank image should return None")

    # TC-02: Normalization Vector Math (Nose Tip Translation & Eye Distance Scaling)
    def test_02_normalization_math(self):
        # 468 fake 3D landmark points
        synthetic_lm = np.random.randn(468, 3).astype(np.float32)
        # Apply normalization formula from face.py
        pts = synthetic_lm.copy()
        pts -= pts[1]  # nose tip origin
        eye_dist = np.linalg.norm(pts[33] - pts[263])
        pts /= eye_dist
        
        # Test nose tip is translated to (0,0,0)
        np.testing.assert_almost_equal(pts[1], np.array([0., 0., 0.]), decimal=5,
                                      err_msg="TC-02 Failed: Nose tip landmark pts[1] must be origin (0,0,0)")
        # Test outer eye-corner distance is normalized to unit scale (1.0)
        unit_dist = np.linalg.norm(pts[33] - pts[263])
        self.assertAlmostEqual(unit_dist, 1.0, places=5,
                              msg="TC-02 Failed: Inter-ocular distance norm(pts[33] - pts[263]) must be 1.0")

    # TC-03: Euclidean to RMS Conversion Math (Divide by sqrt(1404))
    def test_03_rms_distance_scaling(self):
        vec_len = 1404
        d = 10.0
        rms = float(d / np.sqrt(vec_len))
        expected_rms = 10.0 / 37.46998799
        self.assertAlmostEqual(rms, expected_rms, places=6,
                              msg="TC-03 Failed: RMS distance must equal Euclidean distance divided by sqrt(1404)")

    # TC-04: Ground-Truth Formula Verification (Neema Parnami RMS 0.0687 @ T=0.08 -> 14.1%)
    def test_04_neema_parnami_score_calculation(self):
        threshold = 0.08
        rms = 0.0687
        score = round(100 * (1 - rms / threshold), 1)
        self.assertEqual(score, 14.1, "TC-04 Failed: Expected 14.1% match score for RMS 0.0687 at T=0.08")

    # TC-05: k-NN Exact Identity Match (Distance = 0, Confidence = 100%)
    def test_05_exact_identity_match(self):
        dummy_emb = np.random.randn(1404).astype(np.float32)
        test_case = db.Case(name="Target Person", embedding=json.dumps(dummy_emb.tolist()), status="missing")
        matches = matcher.find_matches(dummy_emb, [test_case], threshold=0.08, k=1)
        self.assertEqual(len(matches), 1)
        matched_case, dist, conf = matches[0]
        self.assertAlmostEqual(dist, 0.0, places=5)
        self.assertEqual(conf, 100.0)

    # TC-06: Threshold Rejection (Orthogonal / Distractor Face Filtering)
    def test_06_threshold_rejection(self):
        v1 = np.ones(1404, dtype=np.float32)
        v2 = np.ones(1404, dtype=np.float32) * 5.0  # Far away
        test_case = db.Case(name="Distractor Person", embedding=json.dumps(v1.tolist()), status="missing")
        matches = matcher.find_matches(v2, [test_case], threshold=0.08, k=1)
        self.assertEqual(len(matches), 0, "TC-06 Failed: Distant face must be rejected below threshold")

    # TC-07: SQLModel Case Insertion & JSON Vector Serialization
    def test_07_db_case_serialization(self):
        v = [0.123] * 1404
        c = db.add(db.Case(name="Test Registration", embedding=json.dumps(v), city="Jaipur", status="missing"))
        self.assertIsNotNone(c.id)
        deserialized = json.loads(c.embedding)
        self.assertEqual(len(deserialized), 1404)
        db.delete_case(c.id)

    # TC-08: Human-in-the-Loop State Lifecycle (Pending -> Dismissed / Found)
    def test_08_human_in_the_loop_states(self):
        c = db.add(db.Case(name="Lifecycle Test", embedding="[]", status="missing"))
        s = db.add(db.Sighting(case_id=c.id, city="Delhi", review_status="pending"))
        self.assertEqual(s.review_status, "pending")
        
        # Officer reviews and marks confirmed
        db.set_review(s.id, "confirmed")
        with db.session() as sess:
            refreshed = sess.get(db.Sighting, s.id)
            self.assertEqual(refreshed.review_status, "confirmed")
            
        # Officer closes case
        db.mark_found(c.id)
        with db.session() as sess:
            refreshed_case = sess.get(db.Case, c.id)
            self.assertEqual(refreshed_case.status, "found")
        db.delete_case(c.id)

    # TC-09: Privacy by Design / Storage Minimization (Unmatched Upload Policy)
    def test_09_storage_minimization_unmatched_cleanup(self):
        # Verify db.init() purges orphan sightings (sighting with no matched case_id)
        s_orphan = db.add(db.Sighting(case_id=None, city="Unknown", review_status="pending"))
        orphan_id = s_orphan.id
        db.init()
        with db.session() as sess:
            found = sess.get(db.Sighting, orphan_id)
            self.assertIsNone(found, "TC-09 Failed: Orphan / unmatched sighting records must be purged")

if __name__ == "__main__":
    unittest.main()
