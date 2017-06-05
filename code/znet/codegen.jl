module CodeGen
export timeit

function timeit(s::String, iter=1, title="")
    ret  = ("{\n")
    ret *= ("    auto begin = std::chrono::high_resolution_clock::now();\n")
    ret *= ("for (int i = 0; i < $(iter); i++) {\n")

    ret *= (s)
    ret *= ("}\n")

    ret *= ("    auto end = std::chrono::high_resolution_clock::now();\n")
    ret *= ("    auto duration = std::chrono::duration_cast<std::chrono::microseconds>\
            (end - begin).count();\n")
    ret *= ("    double secs = static_cast<double>(duration) / 1000000;\n")
    ret *= ("    std::cout << \"$(title)\" << secs/$(iter) << \"\\n\";\n")
    ret *= ("}\n")
    return ret
end


end
