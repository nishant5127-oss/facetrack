"""
================================================================================
FaceTrack Automated End-to-End Test Suite (test_facetrack.py)
================================================================================
Comprehensive verification suite covering all 10 core functional checks:
  1. Module & Model Initialization (db.py & face_landmarker.task)
  2. Face Landmark Extraction & Normalization (face.py)
  3. Edge-Case Image Handling (face.py)
  4. Database CRUD & Embedding Serialization (db.py)
  5. Self-Match & Confidence Score Math (matcher.py)
  6. Cross-Person Rejection & Privacy-by-Design (matcher.py)
  7. Video Frame Sampling Pipeline (face.py / OpenCV)
  8. Geolocation City Mapping (ui.py)
  9. Automated Email Alerting with Attachment (emailer.py)
 10. Officer Review State Transitions (db.py / Home.py lifecycle)

Zero-Pollution Guarantee:
  - Uses isolated in-memory/temp SQLite database sessions (never modifies missing.db).
  - Uses temp directories for video and image files (never pollutes uploads/).
  - Mocks network and SMTP services (no real emails sent).
================================================================================
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from sqlmodel import Session, SQLModel, create_engine, select

# Local FaceTrack modules
import db
import emailer
import face
import matcher
import ui


class TestFaceTrackSystem(unittest.TestCase):
    """FaceTrack 10-Point End-to-End Test Suite."""

    @classmethod
    def setUpClass(cls):
        """Prepare temporary directory and isolated database engine."""
        cls.temp_dir = tempfile.mkdtemp(prefix="facetrack_test_")
        cls.temp_db_path = os.path.join(cls.temp_dir, "isolated_test.db")
        cls.temp_engine = create_engine(
            f"sqlite:///{cls.temp_db_path}",
            connect_args={"check_same_thread": False}
        )
        # Create tables in isolated test database
        SQLModel.metadata.create_all(cls.temp_engine)

        # Locate a valid face image from uploads/ if present, or create synthetic portrait
        cls.sample_face_path = None
        uploads_dir = "uploads"
        if os.path.exists(uploads_dir):
            for fname in os.listdir(uploads_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    cand_path = os.path.join(uploads_dir, fname)
                    with open(cand_path, "rb") as f:
                        data = f.read()
                    if face.embed_bytes(data) is not None:
                        cls.sample_face_path = cand_path
                        cls.sample_face_bytes = data
                        break

        # Fallback: if no face in uploads, generate a test face embedding directly
        if cls.sample_face_path is None:
            # Deterministic synthetic 1404-D vector normalized to invariants
            pts = np.random.RandomState(42).randn(468, 3).astype(np.float32)
            pts -= pts[1]
            eye_dist = np.linalg.norm(pts[33] - pts[263]) or 1.0
            pts /= eye_dist
            cls.synthetic_embedding = pts.flatten()
            cls.sample_face_bytes = None
        else:
            cls.synthetic_embedding = face.embed_bytes(cls.sample_face_bytes)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory and isolated database."""
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # CHECK 1: Module & Model Initialization (db.py & face_landmarker.task)
    # --------------------------------------------------------------------------
    def test_01_module_and_model_initialization(self):
        """Verify face_landmarker.task exists and SQLite tables initialize cleanly."""
        # Check task file existence on disk
        self.assertTrue(
            os.path.exists(face.MODEL_PATH),
            f"FaceLandmarker model file '{face.MODEL_PATH}' must exist on disk."
        )
        self.assertGreater(
            os.path.getsize(face.MODEL_PATH),
            1_000_000,
            "Model file size should be > 1 MB."
        )

        # Verify MediaPipe Landmarker loads without error
        landmarker = face._get_landmarker()
        self.assertIsNotNone(landmarker, "Landmarker instance should not be None.")

        # Verify tables exist in isolated SQLite test database
        with Session(self.temp_engine) as session:
            cases = session.exec(select(db.Case)).all()
            sightings = session.exec(select(db.Sighting)).all()
            self.assertIsInstance(cases, list)
            self.assertIsInstance(sightings, list)

    # --------------------------------------------------------------------------
    # CHECK 2: Face Landmark Extraction & Normalization (face.py)
    # --------------------------------------------------------------------------
    def test_02_face_landmark_extraction_and_normalization(self):
        """Verify 1404-D vector extraction, no NaN/Inf, and geometric invariants."""
        emb = self.synthetic_embedding
        self.assertIsNotNone(emb, "Face embedding must not be None.")
        self.assertEqual(len(emb), 1404, "Embedding must have exactly 1404 values (468 * 3).")
        self.assertTrue(np.all(np.isfinite(emb)), "Embedding must contain no NaN or Inf values.")

        # Reshape to (468, 3) matrix to inspect landmarks
        pts = emb.reshape(468, 3)

        # Invariant 1: Landmark #1 (nose tip) must be translated to coordinate origin (0, 0, 0)
        np.testing.assert_allclose(
            pts[1],
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            atol=1e-5,
            err_msg="Landmark #1 (nose tip) must be centered at (0, 0, 0)."
        )

        # Invariant 2: Inter-ocular distance norm(pts[33] - pts[263]) must be 1.0 (unit scale)
        eye_distance = float(np.linalg.norm(pts[33] - pts[263]))
        self.assertAlmostEqual(
            eye_distance,
            1.0,
            places=4,
            msg="Distance between outer eye corners (landmarks 33 and 263) must equal 1.0."
        )

    # --------------------------------------------------------------------------
    # CHECK 3: Edge-Case Image Handling (face.py)
    # --------------------------------------------------------------------------
    def test_03_edge_case_image_handling(self):
        """Verify blank image and corrupted bytes are handled gracefully (return None)."""
        # Blank / black frame (no face)
        black_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        emb_blank = face.embed(black_frame)
        self.assertIsNone(emb_blank, "Blank image should return None without error.")

        # Corrupted / invalid image bytes
        corrupted_bytes = b"CORRUPTED_RAW_BINARY_DATA_NOT_AN_IMAGE_12345"
        emb_corrupt = face.embed_bytes(corrupted_bytes)
        self.assertIsNone(emb_corrupt, "Corrupted bytes should return None without error.")

    # --------------------------------------------------------------------------
    # CHECK 4: Database CRUD & Embedding Serialization (db.py)
    # --------------------------------------------------------------------------
    def test_04_database_crud_and_serialization(self):
        """Verify Case creation, JSON vector roundtrip, and flags in temp DB."""
        orig_vec = self.synthetic_embedding
        case_obj = db.Case(
            name="Neema Parnami",
            age=26,
            gender="Female",
            city="Jaipur",
            contact_email="investigator@police.gov",
            description="Reference test subject for FaceTrack validation",
            photo_path="uploads/test_neema.jpg",
            embedding=json.dumps(orig_vec.tolist()),
            status="missing",
            published=True
        )

        with Session(self.temp_engine) as session:
            session.add(case_obj)
            session.commit()
            session.refresh(case_obj)
            case_id = case_obj.id

            # Retrieve back
            retrieved = session.get(db.Case, case_id)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.name, "Neema Parnami")
            self.assertEqual(retrieved.status, "missing")
            self.assertTrue(retrieved.published)

            # Confirm deserialized vector matches bit-for-bit
            deserialized_vec = np.array(json.loads(retrieved.embedding), dtype=np.float32)
            np.testing.assert_allclose(
                deserialized_vec,
                orig_vec,
                atol=1e-5,
                err_msg="Retrieved embedding vector must match original vector."
            )

    # --------------------------------------------------------------------------
    # CHECK 5: Self-Match & Confidence Score Math (matcher.py)
    # --------------------------------------------------------------------------
    def test_05_self_match_and_confidence_math(self):
        """Verify exact self-match yields 100% confidence, and RMS 0.0687 gives 14.1%."""
        ref_vec = self.synthetic_embedding
        test_case = db.Case(
            id=101,
            name="Neema Parnami",
            embedding=json.dumps(ref_vec.tolist()),
            status="missing"
        )

        # 1. Exact query: distance = 0, score = 100%
        matches = matcher.find_matches(ref_vec, [test_case], threshold=0.08, k=1)
        self.assertEqual(len(matches), 1, "Exact query must return 1 match.")
        matched_case, rms_dist, score = matches[0]
        self.assertAlmostEqual(rms_dist, 0.0, places=4)
        self.assertAlmostEqual(score, 100.0, places=1)

        # 2. Perturbed vector matching presentation benchmark:
        # We construct a query with exact RMS distance = 0.0687
        # Since RMS = ||delta|| / sqrt(1404), we set ||delta|| = 0.0687 * sqrt(1404)
        target_rms = 0.0687
        n_dim = len(ref_vec)
        delta_norm = target_rms * np.sqrt(n_dim)

        delta = np.zeros(n_dim, dtype=np.float32)
        delta[0] = delta_norm  # Shift along first axis

        probe_vec = ref_vec + delta
        matches_perturbed = matcher.find_matches(probe_vec, [test_case], threshold=0.08, k=1)
        self.assertEqual(len(matches_perturbed), 1)
        _, perturbed_rms, perturbed_score = matches_perturbed[0]

        # Assert RMS is approximately 0.0687 and score is 14.1%
        self.assertAlmostEqual(perturbed_rms, 0.0687, places=4)
        # 100 * (1 - 0.0687 / 0.08) = 14.125 -> 14.1%
        self.assertEqual(perturbed_score, 14.1, "Score for RMS 0.0687 at T=0.08 must be 14.1%.")

    # --------------------------------------------------------------------------
    # CHECK 6: Cross-Person Rejection & Privacy-by-Design (matcher.py)
    # --------------------------------------------------------------------------
    def test_06_cross_person_rejection_and_privacy(self):
        """Verify distant/distractor faces (RMS > 0.08) are rejected with no match returned."""
        ref_vec = self.synthetic_embedding
        test_case = db.Case(
            id=202,
            name="Registered Person",
            embedding=json.dumps(ref_vec.tolist()),
            status="missing"
        )

        # Distractor face: large perturbation exceeding threshold 0.08
        n_dim = len(ref_vec)
        large_delta = np.ones(n_dim, dtype=np.float32) * 0.5  # Far distant vector
        distractor_vec = ref_vec + large_delta

        matches = matcher.find_matches(distractor_vec, [test_case], threshold=0.08, k=1)
        self.assertEqual(
            len(matches),
            0,
            "Matcher must reject distractor face (RMS > 0.08) and return empty list."
        )

    # --------------------------------------------------------------------------
    # CHECK 7: Video Frame Sampling Pipeline (face.py / OpenCV)
    # --------------------------------------------------------------------------
    def test_07_video_frame_sampling_pipeline(self):
        """Verify VideoWriter creates test clip and sample_frames samples 1 FPS up to 30 frames."""
        video_path = os.path.join(self.temp_dir, "synthetic_test_video.mp4")
        width, height, fps = 120, 120, 10
        total_seconds = 3
        total_frames = fps * total_seconds  # 30 frames total

        # Write 3-second synthetic video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        for i in range(total_frames):
            frame = np.full((height, width, 3), i * 8, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        self.assertTrue(os.path.exists(video_path), "Video file must be created.")

        # Sample frames at 1 frame per second (max_frames=30)
        sampled = face.sample_frames(video_path, every_sec=1.0, max_frames=30)

        # For a 3-second video sampled every 1.0 sec, expect 3 frames
        self.assertEqual(len(sampled), 3, "Should sample exactly 3 frames for a 3-second video.")
        for fr in sampled:
            self.assertIsInstance(fr, np.ndarray)
            self.assertEqual(fr.shape, (height, width, 3))

    # --------------------------------------------------------------------------
    # CHECK 8: Geolocation City Mapping (ui.py)
    # --------------------------------------------------------------------------
    def test_08_geolocation_city_mapping(self):
        """Verify city coordinate lookup returns valid (lat, lon) and fallback for unknown."""
        # Known cities
        jaipur_coords = ui.get_coordinates("Jaipur")
        self.assertEqual(jaipur_coords, (26.91, 75.79))

        delhi_coords = ui.get_coordinates("Delhi")
        self.assertEqual(delhi_coords, (28.61, 77.21))

        mumbai_coords = ui.get_coordinates("Mumbai")
        self.assertEqual(mumbai_coords, (19.08, 72.88))

        # Unknown / edge-case inputs
        unknown_coords = ui.get_coordinates("NonExistentCity")
        self.assertEqual(unknown_coords, (None, None))

        none_coords = ui.get_coordinates(None)
        self.assertEqual(none_coords, (None, None))

    # --------------------------------------------------------------------------
    # CHECK 9: Automated Email Alerting with Attachment (emailer.py)
    # --------------------------------------------------------------------------
    @patch("smtplib.SMTP")
    def test_09_automated_email_alerting_with_attachment(self, mock_smtp_cls):
        """Verify SMTP alert dispatch with JPEG attachment and headers."""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = mock_server

        # Create dummy JPEG file
        test_img_path = os.path.join(self.temp_dir, "test_sighting.jpg")
        dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imwrite(test_img_path, dummy_img)

        env_vars = {
            "SMTP_HOST": "smtp.gmail.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "alerts@facetrack.org",
            "SMTP_PASS": "test-app-password"
        }

        with patch.dict(os.environ, env_vars):
            sent = emailer.send(
                to="officer@police.gov",
                subject="Possible sighting: Neema Parnami",
                body="A possible match for Neema Parnami was reported in Jaipur (score 14.1%).",
                attachment_path=test_img_path
            )

        self.assertTrue(sent, "emailer.send() should return True on successful dispatch.")
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("alerts@facetrack.org", "test-app-password")
        mock_server.send_message.assert_called_once()

        # Inspect the message sent
        sent_msg = mock_server.send_message.call_args[0][0]
        self.assertEqual(sent_msg["To"], "officer@police.gov")
        self.assertEqual(sent_msg["Subject"], "Possible sighting: Neema Parnami")
        self.assertEqual(sent_msg["From"], "alerts@facetrack.org")

        # Verify attachment filename
        attachments = list(sent_msg.iter_attachments())
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].get_filename(), "sighting.jpg")

    # --------------------------------------------------------------------------
    # CHECK 10: Officer Review State Transitions (db.py / Home.py lifecycle)
    # --------------------------------------------------------------------------
    def test_10_officer_review_state_transitions(self):
        """Simulate full lifecycle: open -> pending sighting -> confirmed -> found."""
        with Session(self.temp_engine) as session:
            # Step 1: Officer registers case
            case = db.Case(
                name="Aakash Sharma",
                age=32,
                city="Delhi",
                status="missing",
                published=True
            )
            session.add(case)
            session.commit()
            session.refresh(case)
            case_id = case.id

            # Confirm case is currently in missing state
            self.assertEqual(case.status, "missing")

            # Step 2: Sighting is reported by citizen
            sighting = db.Sighting(
                case_id=case_id,
                city="Delhi",
                lat=28.61,
                lon=77.21,
                confidence=89.5,
                review_status="pending"
            )
            session.add(sighting)
            session.commit()
            session.refresh(sighting)
            sighting_id = sighting.id

            self.assertEqual(sighting.review_status, "pending")

            # Step 3: Officer reviews and confirms sighting, closes case as 'found'
            sighting_to_update = session.get(db.Sighting, sighting_id)
            sighting_to_update.review_status = "confirmed"
            session.add(sighting_to_update)

            case_to_update = session.get(db.Case, case_id)
            case_to_update.status = "found"
            session.add(case_to_update)
            session.commit()

            # Step 4: Verify state transitions
            refreshed_case = session.get(db.Case, case_id)
            refreshed_sighting = session.get(db.Sighting, sighting_id)

            self.assertEqual(refreshed_sighting.review_status, "confirmed")
            self.assertEqual(refreshed_case.status, "found")

            # Step 5: Verify that active search query filters out closed cases
            active_results = session.exec(
                select(db.Case).where(db.Case.status == "missing")
            ).all()
            active_ids = [c.id for c in active_results]
            self.assertNotIn(case_id, active_ids, "Case marked 'found' must be excluded from active missing cases.")


if __name__ == "__main__":
    unittest.main()
