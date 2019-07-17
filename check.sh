#!/usr/bin/env bash

echo "[ run black ]"
black -v rtransformer/

echo "[ run flake8 ]"
flake8 rtransformer/

echo "[ run isort ]"
isort -rc rtransformer/