import pandas as pd

# Function to process the log file and convert it into a .dat file for LaTeX
def create_dat_file_from_log(log_file, output_file):
    # Read the log file
    with open(log_file, 'r') as file:
        lines = file.readlines()
    lines = lines[1:]
    # Process lines and extract values
    epochs = []
    values = []
    
    for i, line in enumerate(lines, 1):  # Start epoch from 1 and increment for each line
        # Split the line by comma
        parts = line.strip().split(',')
        
        if len(parts) >= 3:
            # epoch is now simply the line number (starting from 1)
            epoch = i
            print(parts[2])
            value = float(parts[2])  # Value (from the third column)
            
            # Append the epoch and value to their respective lists
            epochs.append(epoch)
            values.append(value)

    # Create a DataFrame for easy handling and output
    df = pd.DataFrame({'Epoch': epochs, 'Value': values})

    # Save to .dat file (tab-separated for LaTeX compatibility)
    df.to_csv(output_file, sep=" ", index=False, header=False)

    print(f"Data saved to {output_file}")
# Create the .dat file
create_dat_file_from_log("logData/gru_50_1_64_64_0.15_0.4_version_0.csv", "gru_50_1_64_64_0.15_0.4.dat")
# create_dat_file_from_log("logData/ffnn_[34, 256, 128, 32]_0.4_relu_version_0.csv", "fnn_[34,256,128,32].dat")
# create_dat_file_from_log("logData/ffnn_[34, 256, 128, 64, 32]_0.4_relu_4096_bn_version_0.csv", "fnn_[34,256,128,64,32].dat")
# create_dat_file_from_log("logData/ffnn_[34, 512, 128, 64, 32]_0.4_relu_version_0.csv", "fnn_[34,512,128,64,32].dat")







