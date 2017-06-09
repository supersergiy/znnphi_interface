def indent_lines(lines, indent):
    for i in range(indent):
	lines = list(map(lambda c: "\t" + c, lines))
    return lines

def timeit(code_lines, n=1, title=""):
    lines = []
    lines.append("{")
    lines.append("  auto begin = std::chrono::high_resolution_clock::now()")
    lines.append("  for (int i = 0; i < {}; i++) {".format(n))
    lines += code_lines
    lines.append("    auto end = std::chrono::high_resolution_clock::now();")
    lines.append("    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();")
    lines.append("    double secs = static_cast<double>(duration) / 1000000;")
    lines.append("    std::cout << \"{}\" << secs/{} << \"\\n\";".format(title, n))
    lines.append("}")

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

def fill_tensor(tname, values):
    lines = []
    values_str  = ", ".join(map(str, list(values)))
    alloc_array = 'float {}_init[] = {{}};'.format(tname, values_str)
    lines.append(alloc_array)
    return lines

def zero_out_tensor(tname):
    lines = []
    lines.append('tensors["{}"]->set_to_const(0);'.format(tname))
    return lines


