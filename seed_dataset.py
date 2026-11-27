"""
Seed Kaggle Face Dataset into the Missing Person Tracking System.

This script imports face images from a Kaggle dataset folder (such as
the 31-celebrity Face Recognition dataset) into `missing.db`, extracts
1404-dimensional MediaPipe facial embeddings, stores photos in `uploads/`,
and optionally registers probe sightings or runs matching benchmarks.
"""

import argparse
import json
import os
import random
import shutil
import sys
import uuid
from pathlib import Path

# Local imports
import db
import face
import matcher

DEFAULT_LOCAL_DATASET = Path(r"c:\Users\BIT\Desktop\facetrack\Dataset\Faces")

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Kolkata", "Chennai", "Hyderabad", "Jaipur", "Ranchi", "Pune", "Lucknow"]

KNOWN_METADATA = {
    "Akshay Kumar": {"gender": "Male", "age": 56, "city": "Mumbai"},
    "Alexandra Daddario": {"gender": "Female", "age": 38, "city": "Delhi"},
    "Alia Bhatt": {"gender": "Female", "age": 31, "city": "Mumbai"},
    "Amitabh Bachchan": {"gender": "Male", "age": 81, "city": "Kolkata"},
    "Andy Samberg": {"gender": "Male", "age": 45, "city": "Bengaluru"},
    "Anushka Sharma": {"gender": "Female", "age": 36, "city": "Mumbai"},
    "Billie Eilish": {"gender": "Female", "age": 22, "city": "Jaipur"},
    "Brad Pitt": {"gender": "Male", "age": 60, "city": "Bengaluru"},
    "Camila Cabello": {"gender": "Female", "age": 27, "city": "Kolkata"},
    "Charlize Theron": {"gender": "Female", "age": 48, "city": "Pune"},
    "Claire Holt": {"gender": "Female", "age": 36, "city": "Chennai"},
    "Courtney Cox": {"gender": "Female", "age": 60, "city": "Mumbai"},
    "Dwayne Johnson": {"gender": "Male", "age": 52, "city": "Hyderabad"},
    "Elizabeth Olsen": {"gender": "Female", "age": 35, "city": "Pune"},
    "Ellen Degeneres": {"gender": "Female", "age": 66, "city": "Delhi"},
    "Henry Cavill": {"gender": "Male", "age": 41, "city": "Chennai"},
    "Hrithik Roshan": {"gender": "Male", "age": 50, "city": "Lucknow"},
    "Hugh Jackman": {"gender": "Male", "age": 55, "city": "Ranchi"},
    "Jessica Alba": {"gender": "Female", "age": 43, "city": "Delhi"},
    "Lisa Kudrow": {"gender": "Female", "age": 61, "city": "Mumbai"},
    "Margot Robbie": {"gender": "Female", "age": 34, "city": "Bengaluru"},
    "Natalie Portman": {"gender": "Female", "age": 43, "city": "Kolkata"},
    "Priyanka Chopra": {"gender": "Female", "age": 42, "city": "Mumbai"},
    "Robert Downey Jr": {"gender": "Male", "age": 59, "city": "Delhi"},
    "Roger Federer": {"gender": "Male", "age": 43, "city": "Hyderabad"},
    "Tom Cruise": {"gender": "Male", "age": 62, "city": "Mumbai"},
    "Vijay Deverakonda": {"gender": "Male", "age": 35, "city": "Hyderabad"},
    "Virat Kohli": {"gender": "Male", "age": 36, "city": "Delhi"},
    "Zac Efron": {"gender": "Male", "age": 37, "city": "Jaipur"},
}

def download_from_kaggle(dataset_slug: str, target_dir: Path):
    """Download dataset from Kaggle using kaggle CLI / API."""
    print(f"[*] Attempting to download dataset '{dataset_slug}' from Kaggle...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        target_dir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_slug, path=str(target_dir), unzip=True)
        print(f"[+] Download and extraction complete to {target_dir}")
        return True
    except Exception as e:
        print(f"[-] Kaggle download error: {e}")
        print("    Make sure you have ~/.kaggle/kaggle.json installed with your API key,")
        print("    or download the dataset directly from kaggle.com and supply --data-dir.")
        return False

def seed_dataset(data_dir: Path, limit: int = 15, publish: bool = True, seed_sightings: bool = True):
    db.init()
    os.makedirs("uploads", exist_ok=True)

    if not data_dir.exists():
        print(f"[-] Dataset path does not exist: {data_dir}")
        return

    # Find person directories
    person_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not person_dirs:
        # maybe flat directory
        print("[-] No person subfolders found in", data_dir)
        return

    print(f"[*] Found {len(person_dirs)} person folders in {data_dir}")
    if limit and limit > 0:
        person_dirs = person_dirs[:limit]
        print(f"[*] Processing first {len(person_dirs)} persons (use --limit 0 for all)...")

    registered = []
    probes = []  # (person_name, second_image_path)

    for pdir in person_dirs:
        name = pdir.name
        img_files = sorted(list(pdir.glob("*.jpg")) + list(pdir.glob("*.jpeg")) + list(pdir.glob("*.png")))
        if not img_files:
            continue

        ref_img = img_files[0]
        data = ref_img.read_bytes()
        emb = face.embed_bytes(data)
        if emb is None:
            print(f"  [!] No face detected in reference photo for '{name}', trying next photo...")
            for alt_img in img_files[1:4]:
                data = alt_img.read_bytes()
                emb = face.embed_bytes(data)
                if emb is not None:
                    ref_img = alt_img
                    break

        if emb is None:
            print(f"  [x] Skipped '{name}': No clear front face detected in reference images.")
            continue

        # Save to uploads
        dest_filename = f"uploads/{uuid.uuid4().hex}.jpg"
        with open(dest_filename, "wb") as f:
            f.write(data)

        meta = KNOWN_METADATA.get(name, {
            "gender": random.choice(["Male", "Female"]),
            "age": random.randint(18, 65),
            "city": random.choice(CITIES),
        })

        case = db.Case(
            name=name,
            age=meta.get("age", 30),
            gender=meta.get("gender", ""),
            city=meta.get("city", random.choice(CITIES)),
            contact_email=f"police.desk.{name.lower().replace(' ', '_')}@alerts.gov",
            description=f"Registered missing person record imported from face benchmark dataset ({name}).",
            photo_path=dest_filename,
            embedding=json.dumps(emb.tolist()),
            status="missing",
            published=publish,
        )
        saved_case = db.add(case)
        registered.append(saved_case)
        print(f"  [+] Registered: {name} (ID: {saved_case.id}, City: {saved_case.city}, Age: {saved_case.age})")

        # Keep a second image for probe sighting test
        if len(img_files) > 1:
            probes.append((name, saved_case, img_files[1]))

    print(f"\n[+] Successfully registered {len(registered)} missing person cases in database.")

    # Optionally seed sightings
    if seed_sightings and probes:
        print("\n[*] Seeding sample verified sightings using distinct probe images...")
        sighting_count = 0
        for name, case, probe_path in probes[:5]:  # Add 5 sample sightings
            probe_data = probe_path.read_bytes()
            probe_emb = face.embed_bytes(probe_data)
            if probe_emb is None:
                continue

            # Check match against registered case
            matches = matcher.find_matches(probe_emb, [case], threshold=0.12)
            conf = matches[0][2] if matches else 85.0

            s_dest = f"uploads/{uuid.uuid4().hex}.jpg"
            with open(s_dest, "wb") as f:
                f.write(probe_data)

            sighting = db.Sighting(
                case_id=case.id,
                city=case.city or "Delhi",
                lat=28.61,
                lon=77.21,
                reporter_email="public.witness@citizen.org",
                photo_path=s_dest,
                confidence=conf,
                review_status="pending",
            )
            db.add(sighting)
            sighting_count += 1
            print(f"  [+] Sighting created for {name} (Match confidence: {conf}%)")

        print(f"[+] Added {sighting_count} sample sightings for officer review.")

def run_benchmark(data_dir: Path, threshold: float = 0.08):
    """Run an accuracy check using probe images from each person folder."""
    print("\n" + "=" * 70)
    print(f"[*] Running Facial Recognition Matching Benchmark (Threshold: {threshold})...")
    print("=" * 70)

    cases = db.active_cases()
    if not cases:
        print("[-] No active cases in database. Seed cases first!")
        return

    name_to_case = {c.name: c for c in cases}
    total_tests = 0
    correct_matches = 0

    person_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name in name_to_case]
    print(f"{'Probe Person':<25} | {'Expected Case':<20} | {'Matched Case':<20} | {'RMS Dist':<10} | {'Result'}")
    print("-" * 88)

    for pdir in person_dirs:
        img_files = sorted(list(pdir.glob("*.jpg")) + list(pdir.glob("*.jpeg")) + list(pdir.glob("*.png")))
        if len(img_files) < 2:
            continue

        probe_img = img_files[1]  # Use 2nd image as probe test
        emb = face.embed_bytes(probe_img.read_bytes())
        if emb is None:
            continue

        total_tests += 1
        matches = matcher.find_matches(emb, cases, threshold=threshold, k=1)
        expected = pdir.name
        if matches:
            top_case, dist, conf = matches[0]
            matched_name = top_case.name
            is_correct = (matched_name == expected)
            if is_correct:
                correct_matches += 1
                res = "PASS"
            else:
                res = "MISMATCH"
            print(f"{expected:<25} | {expected:<20} | {matched_name:<20} | {dist:<10.4f} | {res}")
        else:
            print(f"{expected:<25} | {expected:<20} | {'None (below thresh)':<20} | {'--':<10} | MISSED")

    if total_tests > 0:
        acc = (correct_matches / total_tests) * 100
        print("-" * 88)
        print(f"[+] Benchmark Results: {correct_matches}/{total_tests} correct ({acc:.1f}% accuracy)")

def main():
    parser = argparse.ArgumentParser(description="Add Kaggle Face Dataset to Missing Person System")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_LOCAL_DATASET),
                        help="Path to folder containing person subdirectories (default: auto-detected facetrack dataset)")
    parser.add_argument("--kaggle-dataset", type=str, default="",
                        help="Kaggle dataset slug to download (e.g. 'vasukannan/face-recognition-dataset')")
    parser.add_argument("--limit", type=int, default=12,
                        help="Number of individuals to register (default: 12, use 0 for all)")
    parser.add_argument("--publish", action="store_true", default=True,
                        help="Publish cases to public missing persons board (default: True)")
    parser.add_argument("--no-sightings", action="store_true",
                        help="Do not seed sample sightings")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run verification benchmark after seeding")
    parser.add_argument("--threshold", type=float, default=0.085,
                        help="Distance threshold for benchmark (default: 0.085)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.kaggle_dataset:
        dl_dir = Path("kaggle_dataset")
        if download_from_kaggle(args.kaggle_dataset, dl_dir):
            data_dir = dl_dir

    if not data_dir.exists():
        print(f"[-] Directory not found: {data_dir}")
        print("    Provide the path with --data-dir <path>")
        sys.exit(1)

    seed_dataset(
        data_dir=data_dir,
        limit=args.limit,
        publish=args.publish,
        seed_sightings=(not args.no_sightings)
    )

    if args.benchmark:
        run_benchmark(data_dir=data_dir, threshold=args.threshold)

if __name__ == "__main__":
    main()
