#!/bin/bash

tar -xzf ~/new_tests.tar.gz
cp -r ./new_tests/* test/tests/
rm -rf ./new_tests/
rm -rf ~/new_tests.tar.gz
ls ./test/tests
