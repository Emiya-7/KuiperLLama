foreach(required_variable IN ITEMS
        Python3_EXECUTABLE
        QWEN35_FIXTURE_DIR
        QWEN35_FIXTURE_GENERATOR
        QWEN35_EXPORTER
        QWEN35_FIXTURE_MODEL)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "Missing required variable ${required_variable}")
    endif()
endforeach()

file(MAKE_DIRECTORY "${QWEN35_FIXTURE_DIR}")

execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${QWEN35_FIXTURE_GENERATOR}"
            --out_dir "${QWEN35_FIXTURE_DIR}"
    RESULT_VARIABLE generator_status
    OUTPUT_VARIABLE generator_output
    ERROR_VARIABLE generator_error
)
if(NOT generator_status EQUAL 0)
    message(FATAL_ERROR
        "Qwen3.5 tiny fixture generation failed (${generator_status})\n"
        "${generator_output}${generator_error}")
endif()

execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${QWEN35_EXPORTER}"
            --model_dir "${QWEN35_FIXTURE_DIR}"
            --output "${QWEN35_FIXTURE_MODEL}"
            --max_seq_len 128
            --weight_dtype bf16
    RESULT_VARIABLE export_status
    OUTPUT_VARIABLE export_output
    ERROR_VARIABLE export_error
)
if(NOT export_status EQUAL 0)
    message(FATAL_ERROR
        "Qwen3.5 tiny fixture export failed (${export_status})\n"
        "${export_output}${export_error}")
endif()

message(STATUS "${generator_output}")
message(STATUS "${export_output}")
