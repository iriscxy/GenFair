
python write_csv.py $file_path $csv_file_name


cd /ibex/ai/home/chenx0c/code/chexpert-labeler
export PYTHONPATH=/ibex/user/chenx0c/code/NegBio:$PYTHONPATH


python label.py --reports_path /ibex/user/chenx0c/code/V4_pos/chexpert/$csv_file_name --output_path $csv_file_name --verbose

