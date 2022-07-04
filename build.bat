@echo off

if not defined DevEnvDir (
call vc x64
)

set LFLAGS= /link /DYNAMICBASE:NO /NXCOMPAT:NO /out:imp.exe user32.lib kernel32.lib gdi32.lib
set CFLAGS= /MT /Z7 /FC /GS- /Oi /Gm- /GR- /TC /O2 
set SOURCES= ../main.c ../minfft.c

if exist build (
rmdir /S /Q build
mkdir build
pushd build
cl %CFLAGS% %SOURCES% %LFLAGS%
copy ..\alex.jpg .
copy ..\escapi.dll .
.\imp.exe .\alex.jpg
popd build
) else (
mkdir build
pushd build
cl %CFLAGS% %SOURCES% %LFLAGS%
copy ..\alex.jpg .
copy ..\escapi.dll .
.\imp.exe .\alex.jpg
popd build
)
