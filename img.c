#include <windows.h>

const int winx = 500;
const int winy = 500;
const int winbpp = 3;

BITMAPINFO m_bi;
char* buffer = 0;

void create_backbuffer ( int width, int height )
{
    m_bi.bmiHeader.biSize          = sizeof ( BITMAPINFOHEADER );
    m_bi.bmiHeader.biWidth         = width;
    m_bi.bmiHeader.biHeight        = height;
    m_bi.bmiHeader.biPlanes        = 1;
    m_bi.bmiHeader.biBitCount      = 24;
    m_bi.bmiHeader.biCompression   = BI_RGB;
    m_bi.bmiHeader.biSizeImage     = 0;
    m_bi.bmiHeader.biXPelsPerMeter = 100;
    m_bi.bmiHeader.biYPelsPerMeter = 100;
    m_bi.bmiHeader.biClrUsed       = 0;
    m_bi.bmiHeader.biClrImportant  = 0;

    size_t paddedWidth = (winx * 3 + 3) & ~3;
    buffer = new char[paddedWidth * winy * winbpp];

    for(int y = 0; y < winy; ++y)
    {
        for(int x = 0; x < winx; ++x)
        {
            for(int z = 0; z < 3; ++z)
            {
                buffer[y * paddedWidth + x * winbpp + z] = z * x;
            }
        }
    }
}

LRESULT CALLBACK WindowProcedure(HWND, UINT, WPARAM, LPARAM);

char szClassName[] = "CodeBlocksWindowsApp";

int WINAPI WinMain(HINSTANCE hThisInstance,
                   HINSTANCE hPrevInstance,
                   LPSTR lpszArgument,
                   int nCmdShow)
{
    HWND hwnd;               /* This is the handle for our window */
    MSG messages;            /* Here messages to the application are saved */
    WNDCLASSEX wincl;        /* Data structure for the windowclass */

    wincl.hInstance = hThisInstance;
    wincl.lpszClassName = szClassName;
    wincl.lpfnWndProc = WindowProcedure;      /* This function is called by windows */
    wincl.style = CS_DBLCLKS;                 /* Catch double-clicks */
    wincl.cbSize = sizeof (WNDCLASSEX);
    wincl.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wincl.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
    wincl.hCursor = LoadCursor(NULL, IDC_ARROW);
    wincl.lpszMenuName = NULL;                 /* No menu */
    wincl.cbClsExtra = 0;                      /* No extra bytes after the window class */
    wincl.cbWndExtra = 0;                      /* structure or the window instance */
    wincl.hbrBackground = (HBRUSH)COLOR_BACKGROUND;

    if(!RegisterClassEx(&wincl))
        return 0;

    hwnd = CreateWindowEx(
        0,                   /* Extended possibilites for variation */
        szClassName,         /* Classname */
        " project 1 ",       /* Title Text */
        WS_OVERLAPPEDWINDOW, /* default window */
        CW_USEDEFAULT,       /* Windows decides the position */
        CW_USEDEFAULT,       /* where the window ends up on the screen */
        winx,                 /* The programs width */
        winy,                 /* and height in pixels */
        HWND_DESKTOP,        /* The window is a child-window to desktop */
        NULL,                /* No menu */
        hThisInstance,       /* Program Instance handler */
        NULL                 /* No Window Creation data */
        );

    ShowWindow(hwnd, nCmdShow);

    while(GetMessage(&messages, NULL, 0, 0))
    {
        TranslateMessage(&messages);
        DispatchMessage(&messages);
    }

    return messages.wParam;
}

LRESULT CALLBACK WindowProcedure(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    PAINTSTRUCT ps;
    HDC hDC;
    RECT client;
    DWORD result;

    switch(message)                  /* handle the messages */
    {
        case WM_CREATE:
            setbuffer();
            InvalidateRect(hwnd, NULL, TRUE);
            break;

        case WM_PAINT:
            hDC = BeginPaint(hwnd, &ps);
            GetClientRect(hwnd, &client);
            result = StretchDIBits(hDC,
                                   0, 0,
                                   client.right, client.bottom,
                                   0, 0,
                                   winx, winy,
                                   buffer, &m_bi, DIB_RGB_COLORS, SRCCOPY);
            if(result != winy)
            {
                //Drawing failed
                DebugBreak();
            }
            EndPaint(hwnd, &ps);
            break;

        case WM_DESTROY:
            delete buffer;
            PostQuitMessage(0);       /* send a WM_QUIT to the message queue */
            break;
        default:                      /* for messages that we don't deal with */
            return DefWindowProc(hwnd, message, wParam, lParam);
    }
    return 0;
}