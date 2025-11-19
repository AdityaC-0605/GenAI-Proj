#!/bin/bash
# Start overnight training with sleep prevention

cd /Users/aditya/Downloads/Bert_VS_T5

echo "🌙 Starting Overnight Training"
echo ""
echo "This will:"
echo "  • Train mBERT (2-4 hours)"
echo "  • Train mT5 (3-6 hours)"
echo "  • Prevent system sleep"
echo "  • Log everything to logs/overnight_training/"
echo ""
echo "Total time: 5-10 hours"
echo ""
echo "You can check progress with:"
echo "  tail -f logs/overnight_training/training_*.log"
echo ""
echo "Starting in 5 seconds... (Press Ctrl+C to cancel)"
sleep 5

# Start training with caffeinate to prevent sleep
# -i prevents idle sleep
# caffeinate -i runs the command and prevents sleep while it runs
caffeinate -i bash ./train_overnight.sh

echo ""
echo "Training complete! Check logs/overnight_training/ for results."

