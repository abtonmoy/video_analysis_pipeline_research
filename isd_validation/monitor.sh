#!/bin/bash
# Monitor embedding generation progress

echo "Monitoring embedding generation..."
echo ""

while true; do
    COUNT=$(ls /home/wabashcs/abt/video_analysis_pipeline_research-main/isd_validation/results/embeddings/*.npy 2>/dev/null | wc -l)
    echo -ne "\rProgress: $COUNT/500 embeddings generated"
    
    if [ "$COUNT" -ge 500 ]; then
        echo ""
        echo "Complete! All 500 embeddings generated."
        break
    fi
    
    sleep 10
done
