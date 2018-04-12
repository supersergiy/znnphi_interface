import os

def print_tensor_part_lines(tname):
    lines = []
    lines.append('for (int i = 0; i < 5; i++) {{'.format(tname))
    lines.append('  if (i % SIMD_WIDTH == 0) {')
    lines.append('      cout << "|"  << " ";'.format(tname))
    lines.append('  }')
    lines.append('  cout << tensors["{}"]->data()[i] << " ";'.format(tname))
    lines.append('}')
    lines.append('std::cout << std::endl;')
    lines.append('')
    return lines


def print_tensor_lines(tname):
    lines = []
    lines.append('for (int i = 0; i < tensors["{}"]->num_elements(); i++) {{'.format(tname))
    lines.append('  if (i % SIMD_WIDTH == 0) {')
    lines.append('      cout << "|"  << " ";'.format(tname))
    lines.append('  }')
    lines.append('  cout << tensors["{}"]->data()[i] << " ";'.format(tname))
    lines.append('}')
    lines.append('std::cout << std::endl;')
    lines.append('')
    return lines


def indent_lines(lines, indent):
    for i in range(indent):
        lines = list(map(lambda c: "\t" + c, lines))
    return lines

def timeit(code_lines, n=1, title=""):
    lines = []
    lines.append("{")
    lines.append("auto begin = std::chrono::high_resolution_clock::now();")
    lines.append("for (int i = 0; i < {}; i++) {{".format(n))
    lines += indent_lines(code_lines, 1)
    lines.append("}")
    lines.append("auto end = std::chrono::high_resolution_clock::now();")
    lines.append("auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();")
    lines.append("double secs = static_cast<double>(duration) / 1000000;")
    lines.append("std::cout << \"{}\" << secs/{} << \"\\n\";".format(title, n))
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


