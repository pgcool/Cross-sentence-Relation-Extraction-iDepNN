1. Add the below files from LDC dataset
ie.formal.training.texts
ie.formal.test.texts
formal-trng.st.key.02oct95
formal-tst.ST.key.09oct95

2. create data.txt and key.txt.
cat ie.formal.training.texts ie.formal.test.texts > data.txt
cat formal-trng.st.key.02oct95 formal-tst.ST.key.09oct95 > key.txt

3. Run code/scripts/preprocessing/muc6/generate_muc6_data.py to produce "muc6_annotated_data.txt"