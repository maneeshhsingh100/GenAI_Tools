# Create the prepare_data.sh script
prepare_data_script = """
#!/bin/bash

DATA_DIR=training_data
SPEECH_DIR=$DATA_DIR/speech_files
TEXT_DIR=$DATA_DIR/text_files
DEST_DIR=data/local

mkdir -p $DEST_DIR

# Create wav.scp
find $SPEECH_DIR -name "*.wav" | while read file; do
  utt_id=$(basename $file .wav)
  echo "$utt_id $file" >> $DEST_DIR/wav.scp
done

# Create text
find $TEXT_DIR -name "*.txt" | while read file; do
  utt_id=$(basename $file .txt)
  text=$(cat $file)
  echo "$utt_id $text" >> $DEST_DIR/text
done

# Create utt2spk and spk2utt
awk '{print $1 " " $1}' $DEST_DIR/wav.scp > $DEST_DIR/utt2spk
cp $DEST_DIR/utt2spk $DEST_DIR/spk2utt
"""

# Write the script to a file
with open('prepare_data.sh', 'w') as f:
    f.write(prepare_data_script)

# Make the script executable
!chmod +x prepare_data.sh

# Run the script
!./prepare_data.sh
