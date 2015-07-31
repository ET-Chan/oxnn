package = "oxnn"
version = "scm-1"

source = {
   url = "file:///home/et/code/eclipse/oxnn-proj/src"
}

description = {
   summary = "oxnn",
   detailed = [[
   ]],
   homepage = "https://github.com/ET-Chan/oxnn",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "moses",
   "nn",
   "cunn"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
   ]],
   install_command = "cd build && $(MAKE) install"
}
