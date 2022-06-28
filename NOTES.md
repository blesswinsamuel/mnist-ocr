## Add files to DVC

```bash
dvc init
dvc add data/**/*.gz

dvc remote add -d azure azure://mnist-ocr/data
dvc remote modify --local azure account_name mnistocr
dvc remote modify --local azure account_key xxx

dvc push
```
