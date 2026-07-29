#include <tiffio.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
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

struct Image {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> pixels;
};

bool readFloatImage(const std::string& path, Image& image)
{
    std::unique_ptr<TIFF, TiffCloser> tif(TIFFOpen(path.c_str(), "r"));
    if (!tif) {
        return false;
    }

    std::uint16_t bits = 0;
    std::uint16_t samples = 0;
    std::uint16_t format = 0;
    TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &image.width);
    TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &image.height);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &samples);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &format);
    if (bits != 32 || samples != 1 || format != SAMPLEFORMAT_IEEEFP) {
        std::cerr << path << ": expected one-channel float32 TIFF\n";
        return false;
    }

    image.pixels.assign(
        static_cast<std::size_t>(image.width) * image.height, -1.0F
    );
    if (!TIFFIsTiled(tif.get())) {
        for (std::uint32_t row = 0; row < image.height; ++row) {
            auto* dst =
                image.pixels.data() + static_cast<std::size_t>(row) * image.width;
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
    for (std::uint32_t row = 0; row < image.height; row += tileHeight) {
        for (std::uint32_t col = 0; col < image.width; col += tileWidth) {
            if (TIFFReadTile(tif.get(), tile.data(), col, row, 0, 0) < 0) {
                std::cerr << path << ": failed to read tile at " << row << ','
                          << col << '\n';
                return false;
            }
            const auto copyRows = std::min(tileHeight, image.height - row);
            const auto copyCols = std::min(tileWidth, image.width - col);
            for (std::uint32_t localRow = 0; localRow < copyRows; ++localRow) {
                const auto* src =
                    tile.data() + static_cast<std::size_t>(localRow) * tileWidth;
                auto* dst = image.pixels.data() +
                    static_cast<std::size_t>(row + localRow) * image.width + col;
                std::copy_n(src, copyCols, dst);
            }
        }
    }
    return true;
}

struct Surface {
    Image x;
    Image y;
    Image z;

    bool read(const std::string& directory)
    {
        if (!readFloatImage(directory + "/x.tif", x) ||
            !readFloatImage(directory + "/y.tif", y) ||
            !readFloatImage(directory + "/z.tif", z)) {
            return false;
        }
        return x.width == y.width && x.width == z.width &&
            x.height == y.height && x.height == z.height;
    }

    bool valid(std::uint32_t row, std::uint32_t col) const
    {
        if (row >= x.height || col >= x.width) {
            return false;
        }
        const auto index = static_cast<std::size_t>(row) * x.width + col;
        return x.pixels[index] != -1.0F;
    }

    float value(const Image& image, std::uint32_t row,
                std::uint32_t col) const
    {
        return image.pixels[static_cast<std::size_t>(row) * image.width + col];
    }
};

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "usage: compare_tifxyz BASELINE_DIR CANDIDATE_DIR\n";
        return EXIT_FAILURE;
    }

    Surface baseline;
    Surface candidate;
    if (!baseline.read(argv[1]) || !candidate.read(argv[2])) {
        return EXIT_FAILURE;
    }

    const auto unionRows = std::max(baseline.x.height, candidate.x.height);
    const auto unionCols = std::max(baseline.x.width, candidate.x.width);
    std::uint64_t baselineValid = 0;
    std::uint64_t candidateValid = 0;
    std::uint64_t commonValid = 0;
    std::uint64_t baselineOnly = 0;
    std::uint64_t candidateOnly = 0;
    std::uint64_t changedCommon = 0;
    double displacementSum = 0.0;
    double displacementMax = 0.0;
    std::uint32_t maxRow = 0;
    std::uint32_t maxCol = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> candidateOnlyLocations;

    for (std::uint32_t row = 0; row < unionRows; ++row) {
        for (std::uint32_t col = 0; col < unionCols; ++col) {
            const bool beforeValid = baseline.valid(row, col);
            const bool afterValid = candidate.valid(row, col);
            baselineValid += beforeValid;
            candidateValid += afterValid;
            if (beforeValid && !afterValid) {
                ++baselineOnly;
            } else if (!beforeValid && afterValid) {
                ++candidateOnly;
                candidateOnlyLocations.emplace_back(row, col);
            } else if (beforeValid && afterValid) {
                ++commonValid;
                const double dx = static_cast<double>(
                    candidate.value(candidate.x, row, col) -
                    baseline.value(baseline.x, row, col)
                );
                const double dy = static_cast<double>(
                    candidate.value(candidate.y, row, col) -
                    baseline.value(baseline.y, row, col)
                );
                const double dz = static_cast<double>(
                    candidate.value(candidate.z, row, col) -
                    baseline.value(baseline.z, row, col)
                );
                const double displacement = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (displacement != 0.0) {
                    ++changedCommon;
                    displacementSum += displacement;
                    if (displacement > displacementMax) {
                        displacementMax = displacement;
                        maxRow = row;
                        maxCol = col;
                    }
                }
            }
        }
    }

    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"baseline\": \"" << argv[1] << "\",\n";
    std::cout << "  \"candidate\": \"" << argv[2] << "\",\n";
    std::cout << "  \"baseline_shape\": [" << baseline.x.height << ", "
              << baseline.x.width << "],\n";
    std::cout << "  \"candidate_shape\": [" << candidate.x.height << ", "
              << candidate.x.width << "],\n";
    std::cout << "  \"baseline_valid_vertices\": " << baselineValid << ",\n";
    std::cout << "  \"candidate_valid_vertices\": " << candidateValid << ",\n";
    std::cout << "  \"common_valid_vertices\": " << commonValid << ",\n";
    std::cout << "  \"baseline_only_valid_vertices\": " << baselineOnly << ",\n";
    std::cout << "  \"candidate_only_valid_vertices\": " << candidateOnly
              << ",\n";
    std::cout << "  \"changed_common_xyz_vertices\": " << changedCommon << ",\n";
    std::cout << "  \"changed_common_mean_displacement\": "
              << (changedCommon == 0 ? 0.0 : displacementSum / changedCommon)
              << ",\n";
    std::cout << "  \"changed_common_max_displacement\": " << displacementMax
              << ",\n";
    std::cout << "  \"changed_common_max_location\": [" << maxRow << ", "
              << maxCol << "],\n";
    std::cout << "  \"candidate_only_locations\": [";
    for (std::size_t index = 0; index < candidateOnlyLocations.size(); ++index) {
        if (index != 0) {
            std::cout << ", ";
        }
        std::cout << '[' << candidateOnlyLocations[index].first << ", "
                  << candidateOnlyLocations[index].second << ']';
    }
    std::cout << "]\n";
    std::cout << "}\n";
    return EXIT_SUCCESS;
}
