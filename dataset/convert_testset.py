import json

input_file = '32final_testset.json'
output_file = '32final_testset_nl.json'

new_instruction = "Predict whether the applicant will default on the loan based on the following information. Only answer 'Default' or 'No Default'."

def convert_output(val):
    if val == "1" or val == 1:
        return "Default"
    elif val == "0" or val == 0:
        return "No Default"
    elif val == "":
        return ""
    else:
        return val  # 保留原样

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    item['instruction'] = new_instruction
    if 'output' in item:
        item['output'] = convert_output(item['output'])

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"转换完成，已保存为 {output_file}")
