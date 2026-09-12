include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
aux_source_directory(../test DIR_TEST)

aux_source_directory(../test/test_cu DIR_TEST_CU)
aux_source_directory(../test/test_op DIR_TEST_OP)
aux_source_directory(../test/test_model DIR_TEST_MODEL)
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)
aux_source_directory(../test/optimized DIR_TEST_OPTIMIZED)

add_executable(test_llm ${DIR_TEST} ${DIR_TEST_CU} ${DIR_TEST_OP} ${DIR_TEST_OPTIMIZED} ${DIR_TEST_TENSOR} ${DIR_TEST_MODEL})

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
target_link_libraries(test_llm ${link_ext_lib})
target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ${GTest_INCLUDE_DIR})

target_include_directories(test_llm PUBLIC ../kuiper/include)
target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)
if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT OR QWEN35_SUPPORT)
    message(STATUS "LINK LLAMA3 SUPPORT")
    find_package(absl REQUIRED)
    find_package(re2 REQUIRED)
    find_package(nlohmann_json REQUIRED)
    target_link_libraries(llama absl::base re2::re2 nlohmann_json::nlohmann_json)
endif ()
target_link_libraries(test_llm llama)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_llm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(QWEN35_SUPPORT)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(QWEN35_FIXTURE_DIR ${CMAKE_CURRENT_BINARY_DIR}/fixtures/qwen35)
    set(QWEN35_FIXTURE_MODEL ${QWEN35_FIXTURE_DIR}/tiny-qwen35-bf16.bin)
    set(QWEN35_FIXTURE_TOKENIZER ${QWEN35_FIXTURE_DIR}/tokenizer.json)

    add_test(
        NAME qwen35_tiny_fixture
        COMMAND ${CMAKE_COMMAND}
            -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
            -DQWEN35_FIXTURE_DIR=${QWEN35_FIXTURE_DIR}
            -DQWEN35_FIXTURE_GENERATOR=${CMAKE_CURRENT_SOURCE_DIR}/test_model/make_tiny_qwen35.py
            -DQWEN35_EXPORTER=${CMAKE_SOURCE_DIR}/tools/export_qwen35/export.py
            -DQWEN35_FIXTURE_MODEL=${QWEN35_FIXTURE_MODEL}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/prepare_qwen35_fixture.cmake
    )
    set_tests_properties(qwen35_tiny_fixture PROPERTIES
        FIXTURES_SETUP qwen35_tiny
        LABELS fixture
        TIMEOUT 120
    )
endif()

add_test(NAME test_llm COMMAND $<TARGET_FILE:test_llm>)
set_tests_properties(test_llm PROPERTIES
    LABELS "gpu;qwen35"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RUN_SERIAL TRUE
    TIMEOUT 300
)
if(QWEN35_SUPPORT)
    set_tests_properties(test_llm PROPERTIES
        FIXTURES_REQUIRED qwen35_tiny
        ENVIRONMENT
            "GLOG_logtostderr=1;KUIPER_TINY_QWEN35=${QWEN35_FIXTURE_MODEL};KUIPER_TINY_QWEN35_TOKENIZER=${QWEN35_FIXTURE_TOKENIZER}"
    )
endif()
