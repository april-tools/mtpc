#!/bin/bash

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_file>"
    echo "Example: $0 throughput_commands.sh"
    exit 1
fi

INPUT_FILE="$1"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Read the input file line by line
while IFS= read -r line; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^#.*$ ]]; then
        continue
    fi
    
    # Extract the parameters from the command line
    # Expected format: ./bin/compute_throughput_speculative outputs/models/MODEL MODEL PART STEP
    if [[ "$line" =~ ./bin/compute_throughput_speculative_argmax[[:space:]]+outputs/models/([^[:space:]]+)[[:space:]]+([^[:space:]]+)[[:space:]]+([^[:space:]]+)[[:space:]]+([^[:space:]]+) ]]; then
        model="${BASH_REMATCH[2]}"
        part="${BASH_REMATCH[3]}"
        step="${BASH_REMATCH[4]}"
        
        # Generate the output filename
        output_file="throughput-${model}-${part}-${step}.sh"
        
        # Create the individual script file
        cat > "$output_file" << EOF
#!/bin/bash
$line
EOF
        
        # Make the generated script executable
        chmod +x "$output_file"
        
        echo "Created: $output_file"
    else
        echo "Warning: Skipping line that doesn't match expected format: $line"
    fi
done < "$INPUT_FILE"

echo "Done! Generated individual script files."
