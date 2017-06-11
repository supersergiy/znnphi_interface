import os

def indent_lines(lines, indent):
    for i in range(indent):
	lines = list(map(lambda c: "\t" + c, lines))
    return lines

def timeit(code_lines, n=1, title=""):
    lines = []
    lines.append("auto begin = std::chrono::high_resolution_clock::now();")
    lines.append("for (int i = 0; i < {}; i++) {{".format(n))
    lines += indent_lines(code_lines, 1)
    lines.append("}")
    lines.append("auto end = std::chrono::high_resolution_clock::now();")
    lines.append("auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();")
    lines.append("double secs = static_cast<double>(duration) / 1000000;")
    lines.append("std::cout << \"{}\" << secs/{} << \"\\n\";".format(title, n))

    return lines

def generate_function(signature, body_lines):
    lines = []
    lines.append('')
    lines.append(signature)
    lines.append('{')
    lines += indent_lines(body_lines, 1)
    lines.append('}')
    lines.append('')

    return lines

def write_values_to_file(values, file_name):
    with open(file_name, 'w') as f:
        for v in values:
            f.write("{} ".format(str(v)))

def fill_tensor(tname, values):
    lines = []
    out_directory  = './out/weights'
    data_file_name = '{}.data'.format(tname)
    data_path      = os.path.join(out_directory, data_file_name)
    print "writing to {}!".format(data_path)
    write_values_to_file(values, data_path)
    fill_in_array = 'readArrayFromFile(tensors["{}"]->data(), weights_path + "{}");'.format(tname, data_file_name)

    lines.append(fill_in_array)

    return lines

def zero_out_tensor(tname):
    lines = []
    lines.append('tensors["{}"]->set_to_const(0);'.format(tname))
    return lines


