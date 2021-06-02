#include <iostream>

using namespace std;

int mat_histos() {

    // V A R I A B L E   D E C L A R A T I O N 
    Int_t nbox = 10, len;
    Int_t mgrid[nbox][nbox];


    // Fill matrix mgrid
    for (int i = 0; i < nbox; i++) {
        for (int j = 0; j < nbox; j++) {
            Int_t val = nbox * nbox - (i + j);
            mgrid[i][j] = val;
        }
    }


    // 2 D   H I S T O G R A M
    //
    // Create Canvas (window)
    auto c1    = new TCanvas("c1","c1",600,400);

    // Create 2D Histogram with TH2F
    //                     Name  | Title         | xbins | xmin | xmax | ybins | ymin | ymax
    auto hcol1 = new TH2F("hcol1", "2D Histogram", nbox  , 0    , nbox , nbox  , 0    , nbox);

    // Fill the 2D histogram with mgrid values
    for (int i = 0; i < nbox; i++) {
        for (int j = 0; j < nbox; j++) {
            hcol1 -> Fill(i, j, mgrid[i][j]);
        }
    }

    hcol1->Draw("COLZ");


    // L E G O   H I S T O G R A M
    //
    // Create Canvas (window)
    auto c2    = new TCanvas("c2","c2",600,400);

    // Create 2D Histogram with TH2F
    //                     Name  | Title         | xbins | xmin | xmax | ybins | ymin | ymax
    auto hcol2 = new TH2F("hcol2", "LEGO Histogram", nbox  , 0    , nbox , nbox  , 0    , nbox);

    // Fill the 2D histogram with mgrid values
    for (int i = 0; i < nbox; i++) {
        for (int j = 0; j < nbox; j++) {
            hcol2 -> Fill(i, j, mgrid[i][j]);
        }
    }


    hcol2->Draw("LEGO2Z");  // Lego plot using colors
    // hcol2->Draw("SURF7");  // Like LEGO but with softened lines and height map on top

    return 0;
}
