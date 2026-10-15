catch2_dep = dependency('catch2-with-main', required: true)

srcdir = '../src'
unittest_src = []
foreach src : gamescope_core_src + ['Script/Script.cpp', 'convar_script.cpp']
	unittest_src += srcdir / src
endforeach

test_convar = executable(
	'test_convar',
	unittest_src + ['test_convar.cpp'],
	include_directories: [srcdir, sol2_include],
	dependencies: [catch2_dep, luajit_dep, cap_dep, drm_dep, glm_dep, wlroots_dep],
	cpp_args: gamescope_cpp_args,
)
test('convar', test_convar)

test_framegen = executable(
	'test_framegen',
	['test_framegen.cpp', srcdir / 'framegen/atomic_file.cpp'],
	include_directories: [srcdir],
	cpp_args: gamescope_cpp_args,
)
test('framegen', test_framegen)

test_framegen_deadline = executable(
	'test_framegen_deadline',
	['framegen_deadline.cpp'],
	include_directories: [srcdir],
	dependencies: [catch2_dep],
	cpp_args: gamescope_cpp_args,
)
test('framegen_deadline', test_framegen_deadline)

test_swapchain_policy = executable(
	'test_swapchain_policy',
	['test_swapchain_policy.cpp'],
	include_directories: [srcdir],
	dependencies: [catch2_dep, vulkan_dep.partial_dependency(compile_args: true, includes: true)],
	cpp_args: gamescope_cpp_args,
)
test('swapchain_policy', test_swapchain_policy)

test_vblank_grid = executable(
	'test_vblank_grid',
	['test_vblank_grid.cpp'],
	include_directories: [srcdir],
	dependencies: [catch2_dep],
	cpp_args: gamescope_cpp_args,
)
test('vblank_grid', test_vblank_grid)
