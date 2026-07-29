#include <tiffio.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct TiffCloser {
    void operator()(TIFF* tif) const
    {
        if (tif != nullptr) {
            TIFFClose(tif);
        }
    }
};

bool readFloatImage(const char* path, std::vector<float>& pixels,
                    std::uint32_t& width, std::uint32_t& height)
{
    std::unique_ptr<TIFF, TiffCloser> tif(TIFFOpen(path, "r"));
    if (!tif) {
        return false;
    }

    std::uint16_t bits = 0;
    std::uint16_t samples = 0;
    std::uint16_t format = 0;
    TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &height);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &samples);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &format);
    if (bits != 32 || samples != 1 || format != SAMPLEFORMAT_IEEEFP) {
        std::cerr << path << ": expected one-channel float32 TIFF\n";
        return false;
    }

    pixels.assign(static_cast<std::size_t>(width) * height, -1.0F);
    if (!TIFFIsTiled(tif.get())) {
        for (std::uint32_t row = 0; row < height; ++row) {
            auto* dst = pixels.data() + static_cast<std::size_t>(row) * width;
            if (TIFFReadScanline(tif.get(), dst, row, 0) < 0) {
                std::cerr << path << ": failed to read row " << row << '\n';
                return false;
            }
        }
        return true;
    }

    std::uint32_t tileWidth = 0;
    std::uint32_t tileHeight = 0;
    TIFFGetField(tif.get(), TIFFTAG_TILEWIDTH, &tileWidth);
    TIFFGetField(tif.get(), TIFFTAG_TILELENGTH, &tileHeight);
    std::vector<float> tile(
        static_cast<std::size_t>(tileWidth) * tileHeight
    );
    for (std::uint32_t row = 0; row < height; row += tileHeight) {
        for (std::uint32_t col = 0; col < width; col += tileWidth) {
            if (TIFFReadTile(tif.get(), tile.data(), col, row, 0, 0) < 0) {
                std::cerr << path << ": failed to read tile at " << row << ','
                          << col << '\n';
                return false;
            }
            const auto copyRows = std::min(tileHeight, height - row);
            const auto copyCols = std::min(tileWidth, width - col);
            for (std::uint32_t localRow = 0; localRow < copyRows; ++localRow) {
                const auto* src =
                    tile.data() + static_cast<std::size_t>(localRow) * tileWidth;
                auto* dst = pixels.data() +
                    static_cast<std::size_t>(row + localRow) * width + col;
                std::copy_n(src, copyCols, dst);
            }
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "usage: scan_tifxyz_boundary X_TIF [...]\n";
        return EXIT_FAILURE;
    }

    std::cout
        << "path\trows\tcols\tvalid_vertices\tvalid_quads\tadded_quads"
           "\tlast_row_quads\tlast_col_quads\tcorner_quad"
           "\tadded_quad_origins_row_col\n";
    int failures = 0;
    for (int arg = 1; arg < argc; ++arg) {
        std::vector<float> x;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        if (!readFloatImage(argv[arg], x, width, height)) {
            ++failures;
            continue;
        }

        std::uint64_t validVertices = 0;
        for (const float value : x) {
            validVertices += value != -1.0F;
        }

        std::uint64_t validQuads = 0;
        std::uint64_t lastRowQuads = 0;
        std::uint64_t lastColQuads = 0;
        std::uint64_t cornerQuad = 0;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> addedOrigins;
        if (height >= 2 && width >= 2) {
            for (std::uint32_t row = 0; row + 1 < height; ++row) {
                for (std::uint32_t col = 0; col + 1 < width; ++col) {
                    const auto i =
                        static_cast<std::size_t>(row) * width + col;
                    const bool valid =
                        x[i] != -1.0F && x[i + 1] != -1.0F &&
                        x[i + width] != -1.0F &&
                        x[i + width + 1] != -1.0F;
                    if (!valid) {
                        continue;
                    }
                    ++validQuads;
                    if (row + 2 == height) {
                        ++lastRowQuads;
                    }
                    if (col + 2 == width) {
                        ++lastColQuads;
                    }
                    if (row + 2 == height && col + 2 == width) {
                        ++cornerQuad;
                    }
                    if (row + 2 == height || col + 2 == width) {
                        addedOrigins.emplace_back(row, col);
                    }
                }
            }
        }
        const auto addedQuads =
            lastRowQuads + lastColQuads - cornerQuad;
        std::cout << argv[arg] << '\t' << height << '\t' << width << '\t'
                  << validVertices << '\t' << validQuads << '\t'
                  << addedQuads << '\t' << lastRowQuads << '\t'
                  << lastColQuads << '\t' << cornerQuad << '\t';
        for (std::size_t index = 0; index < addedOrigins.size(); ++index) {
            if (index != 0) {
                std::cout << ';';
            }
            std::cout << addedOrigins[index].first << ','
                      << addedOrigins[index].second;
        }
        std::cout << '\n';
    }
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
