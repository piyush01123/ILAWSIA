"""
Divides train data to Query and Search DB.
"""

import argparse
import random
import os
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="NCT-CRC-HE-100K")
    parser.add_argument("--query_dir", default="QuerySet")
    parser.add_argument("--search_dir", default="SearchSet")
    parser.add_argument("--query_db_size", default=10)
    args = parser.parse_args()

    os.makedirs(args.query_dir, exist_ok=True)
    os.makedirs(args.search_dir, exist_ok=True)

    for className in os.listdir(args.root_dir):
        os.makedirs(os.path.join(args.query_dir, className), exist_ok=True)
        os.makedirs(os.path.join(args.search_dir, className), exist_ok=True)

        allFiles = sorted(os.listdir(os.path.join(args.root_dir, className)))
        random.seed(42)
        queryFiles = random.sample(allFiles, args.query_db_size)
        for file in queryFiles:
            src = os.path.join(args.root_dir,className,file)
            dest = os.path.join(args.query_dir,className)
            shutil.move(src,dest)

        for file in os.listdir(os.path.join(args.root_dir, className)):
            src = os.path.join(args.root_dir,className,file)
            dest = os.path.join(args.search_dir,className)
            shutil.move(src,dest)


if __name__ == "__main__":
    main()
