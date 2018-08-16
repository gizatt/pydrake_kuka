#!/usr/bin/env python
from __future__ import print_function

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--drake_url", type=str,
        default="https://drake-packages.csail.mit.edu/"
                "drake/nightly/drake-latest-xenial.tar.gz",
        help="(optional) drake binary URL to use")

    args = parser.parse_args()

    print("building docker container . . . ")

    print("building docker image named pydrake_kuka")
    print("using drake url %s" % args.drake_url)
    cmd = "docker build -t pydrake_kuka --build-arg DRAKE_URL=%s ." % (
        args.drake_url)

    print("command = \n \n", cmd)
    print("")

    os.system(cmd)
