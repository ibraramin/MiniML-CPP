#ifndef MINIML_SERIALIZATION_HPP
#define MINIML_SERIALIZATION_HPP

// Versioned, endian-safe binary persistence for MiniML-CPP Tensor weights.
//
// Wire format (all integers little-endian, all floats IEEE-754 binary32):
//   char[4]   magic       "MMLW"
//   uint16    version     1
//   uint32    count       number of tensors
//   per tensor:
//     uint32  rank
//     uint32  dim[rank]
//     float32 data[prod(dim)]
//
// No struct is ever dumped: every field is written byte by byte, so files are
// portable across host endianness and alignment. save()/load() return false
// (rather than throwing) on any mismatch, truncation or I/O failure. load()
// validates the whole file before touching any destination tensor.

#include "tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace miniml_serialization {

static constexpr uint16_t kVersion = 1;
static constexpr char kMagic[4] = {'M', 'M', 'L', 'W'};

namespace detail {

inline void put_u16(std::vector<uint8_t> &b, uint16_t v)
{
    b.push_back(static_cast<uint8_t>(v & 0xFFu));
    b.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
}

inline void put_u32(std::vector<uint8_t> &b, uint32_t v)
{
    for (int shift = 0; shift < 32; shift += 8)
        b.push_back(static_cast<uint8_t>((v >> shift) & 0xFFu));
}

inline void put_f32(std::vector<uint8_t> &b, float f)
{
    uint32_t bits = 0;
    static_assert(sizeof(float) == 4, "serialization requires 32-bit float");
    std::memcpy(&bits, &f, sizeof(bits));
    put_u32(b, bits);
}

inline bool get_u16(const std::vector<uint8_t> &b, std::size_t &pos, uint16_t &out)
{
    if (pos + 2 > b.size())
        return false;
    out = static_cast<uint16_t>(b[pos]) |
          static_cast<uint16_t>(static_cast<uint16_t>(b[pos + 1]) << 8);
    pos += 2;
    return true;
}

inline bool get_u32(const std::vector<uint8_t> &b, std::size_t &pos, uint32_t &out)
{
    if (pos + 4 > b.size())
        return false;
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i)
        v |= static_cast<uint32_t>(b[pos + i]) << (8 * i);
    pos += 4;
    out = v;
    return true;
}

inline bool get_f32(const std::vector<uint8_t> &b, std::size_t &pos, float &out)
{
    uint32_t bits = 0;
    if (!get_u32(b, pos, bits))
        return false;
    static_assert(sizeof(float) == 4, "serialization requires 32-bit float");
    std::memcpy(&out, &bits, sizeof(out));
    return true;
}

struct TensorRecord
{
    size_vec shape;
    std::vector<float> values;
};

} // namespace detail

inline bool save(const std::string &path, const std::vector<Tensor *> &weights)
{
    if (path.empty())
        return false;
    for (const Tensor *w : weights)
        if (w == nullptr || w->data == nullptr)
            return false;

    std::vector<uint8_t> b;
    for (char c : kMagic)
        b.push_back(static_cast<uint8_t>(c));
    detail::put_u16(b, kVersion);
    detail::put_u32(b, static_cast<uint32_t>(weights.size()));

    for (const Tensor *w : weights)
    {
        if (w->shape.size() > UINT32_MAX)
            return false;
        detail::put_u32(b, static_cast<uint32_t>(w->shape.size()));
        for (std::size_t d : w->shape)
        {
            if (d > UINT32_MAX)
                return false;
            detail::put_u32(b, static_cast<uint32_t>(d));
        }
        for (std::size_t i = 0; i < w->total_size; ++i)
            detail::put_f32(b, w->data->at(i));
    }

    std::ofstream out(path, std::ios::binary);
    if (!out)
        return false;
    out.write(reinterpret_cast<const char *>(b.data()),
              static_cast<std::streamsize>(b.size()));
    out.flush();
    return out.good();
}

inline bool load(const std::string &path, std::vector<Tensor *> &weights)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    const std::vector<uint8_t> b((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
    if (in.bad())
        return false;

    std::size_t pos = 0;
    for (char c : kMagic)
    {
        if (pos >= b.size() || b[pos++] != static_cast<uint8_t>(c))
            return false; // bad magic
    }

    uint16_t version = 0;
    if (!detail::get_u16(b, pos, version) || version != kVersion)
        return false; // unsupported version

    uint32_t count = 0;
    if (!detail::get_u32(b, pos, count))
        return false;
    if (count != weights.size())
        return false;

    // Parse and validate everything before mutating any destination tensor.
    std::vector<detail::TensorRecord> parsed;
    parsed.reserve(count);
    for (uint32_t k = 0; k < count; ++k)
    {
        uint32_t rank = 0;
        if (!detail::get_u32(b, pos, rank) || rank == 0 || rank > 8)
            return false; // implausible rank / truncated header

        detail::TensorRecord rec;
        rec.shape.reserve(rank);
        std::size_t expected = 1;
        for (uint32_t r = 0; r < rank; ++r)
        {
            uint32_t dim = 0;
            if (!detail::get_u32(b, pos, dim) || dim == 0)
                return false;
            if (expected > SIZE_MAX / dim)
                return false; // product overflow
            expected *= dim;
            rec.shape.push_back(static_cast<std::size_t>(dim));
        }
        if (expected > SIZE_MAX / sizeof(float))
            return false;
        // Subtraction avoids size_t wraparound for crafted huge dimensions:
        // `pos + expected * sizeof(float)` can wrap and pass this guard.
        if (expected > (b.size() - pos) / sizeof(float))
            return false; // truncated payload

        rec.values.reserve(expected);
        for (std::size_t i = 0; i < expected; ++i)
        {
            float f = 0.0f;
            if (!detail::get_f32(b, pos, f))
                return false;
            rec.values.push_back(f);
        }
        parsed.push_back(std::move(rec));
    }

    if (pos != b.size())
        return false; // trailing garbage

    for (std::size_t k = 0; k < parsed.size(); ++k)
    {
        Tensor *w = weights[k];
        if (w == nullptr || w->data == nullptr || w->shape != parsed[k].shape ||
            w->total_size != parsed[k].values.size())
            return false; // shape mismatch with destination
    }

    for (std::size_t k = 0; k < parsed.size(); ++k)
        std::copy(parsed[k].values.begin(), parsed[k].values.end(), weights[k]->data->begin());

    return true;
}

} // namespace miniml_serialization

#endif // MINIML_SERIALIZATION_HPP
