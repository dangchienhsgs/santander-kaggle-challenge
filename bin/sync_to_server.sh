#!/usr/bin/env bash

rsync -avzh . --exclude .git  zdeploy@10.40.19.17:/data/chiennd/projects/santander-value-prediction