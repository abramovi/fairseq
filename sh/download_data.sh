#!/bin/bash

# sudo apt-get update
# sudo apt-get install build-essential libpq-dev libssl-dev openssl libffi-dev zlib1g-dev
# sudo apt-get install python3-pip python3.7-dev
# sudo apt-get install python3.7
# sudo apt autoremove python3
# sudo apt-get install wget

cd ../examples/language_model/
bash prepare-wikitext-103.sh
cd ../..

