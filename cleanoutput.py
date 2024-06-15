import re

def parse_and_format(filepath):
    with open(filepath, 'r') as file:
        data = file.read()
    with open(filepath+"_cln", 'w') as file:
        sections = re.split(r"Finished sentence \d+\n", data)
        
        for i,section in enumerate(sections[1:]):  # Skip the first split as it's empty
            section = section.replace("\\n", " ")
            input_match = re.search(r"Input: \[.{1}\[INST\] (.*?\])", section, re.DOTALL)

            original_output_match = re.search(r"Original Output: (\[.*?\])", section, re.DOTALL)
            pos_output_match = re.search(r"Pos output: (\[.*?\])", section, re.DOTALL)
            neg_output_match = re.search(r"Neg output: (\[.*?\])", section, re.DOTALL)
            zeroedout_output_match = re.search(r"Zeroedout output: (\[.*?\])", section, re.DOTALL)
            
            # Print formatted results
            file.write(f"Input {i+1}:\n")
            file.write(input_match.group(1).replace(" [/INST]","") if input_match else "Not found")
            file.write("\nOriginal Output:\n")
            file.write(original_output_match.group(1)[1:-1] if original_output_match else "Not found")
            file.write("\nPos Steered Output:\n")
            file.write(pos_output_match.group(1)[1:-1] if pos_output_match else "Not found")
            file.write("\nNeg Steered Output:\n")
            file.write(neg_output_match.group(1)[1:-1] if neg_output_match else "Not found")
            file.write("\nZeroed Out Output:\n")
            file.write(zeroedout_output_match.group(1)[1:-1] if zeroedout_output_match else "Not found")
            file.write("\n---\n")

if __name__ == "__main__":
    parse_and_format('output/steering_morallyambiguous_new_unpaired_l14m30.txt')
