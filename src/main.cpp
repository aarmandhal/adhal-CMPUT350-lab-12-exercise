// Lab 12: 2D ray tracing with a BVH. Work through the TODOs below and answer the questions in
// written.txt.
// TODO Bonus 2: some circles are drawn with their sides cut off; fix them.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "aabb.h"
#include "vector4.h"
#include "vector_util.h"
#include <SFML/Graphics.hpp>

constexpr int gScreenWidth = 1024;
constexpr int gScreenHeight = 768;
constexpr float EPSILON = 0.001f;
constexpr bool REBUILD = true;  // true = rebuild BVH every frame (Q4), false = refit only (Q3)

struct Circle {
    constexpr bool intersects(sf::Vector2f point) const {
        return (point - position).lengthSquared() < radius * radius;
    }

    constexpr AABB<float> bounds() const { return {position - radius, position + radius}; }

    sf::Vector2f position;
    float radius;
    Vector4f color;
};

struct BVHNode {
    AABB<float> bounds;
    int childAOrIdx = -1;  // if leaf node, first index into index list, else child bvh node ptr
    int childBOrLen = -1;  // if leaf node, length into index list, else child bvh node ptr
    int isLeaf = 0;
    // Bonus 1: 0 for leaves, 1 + taller child for internal nodes (replaces padding)
    int height = 0;
};
static_assert(sizeof(BVHNode) == 32, "keep BVHNode a power-of-two size");

struct BVH {
    static constexpr float T_ISECT = 1;
    static constexpr float T_TRAV = 1.f / 8;
    static constexpr int BINS = 32;

    explicit BVH(const std::vector<Circle>& prims) : primitives(&prims), indices(prims.size()) {
        std::iota(indices.begin(), indices.end(), 0);  // start with indices 0..n-1
        nodes.reserve(2 * prims.size());               // max nodes, so nodes never reallocates
        reset();                                       // build the tree (same code as rebuilding)
    }

    // Q4: rebuild the whole tree from the circles' current positions
    void reset() {
        nodes.clear();                                          // throw away the old tree
        nodes.push_back({});                                    // fresh empty root at index 0
        constructNode(0, static_cast<int>(indices.size()), 0);  // build the tree from the root
        computeHeights();                                       // Bonus 1: fill in node heights
    }

    // Bonus 1: children are stored after their parent, so a reverse sweep sees children first
    void computeHeights() {
        for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
            BVHNode& node = nodes[i];
            if (node.isLeaf) {
                node.height = 0;  // leaves are at the bottom
            } else {
                node.height =
                    1 + std::max(nodes[node.childAOrIdx].height,
                                 nodes[node.childBOrLen].height);  // one above taller child
            }
        }
    }

    void constructNode(int startIdx, int numIndices, int bvhOffset) {
        const std::vector<Circle>& prims = *primitives;
        const auto first = indices.begin() + startIdx;
        const auto last = first + numIndices;
        const auto centroid = [&](int idx) { return prims[idx].bounds().centroid(); };

        // determine total bound of centroids (to determine bins)
        AABB<float> centroidBound{};
        for (auto it = first; it != last; ++it) {
            centroidBound |= centroid(*it) + EPSILON;
            centroidBound |= centroid(*it) - EPSILON;
        }
        const sf::Vector2f binWidth =
            (centroidBound.maxPoint - centroidBound.minPoint) / static_cast<float>(BINS);
        assert(binWidth.x != 0);
        assert(binWidth.y != 0);
        const sf::Vector2f invBinWidth = {1.f / binWidth.x, 1.f / binWidth.y};

        // find bin each primitive belongs to, per dimension (0 = x, 1 = y)
        std::array<std::array<AABB<float>, BINS>, 2> binBounds{};
        std::array<std::array<int, BINS>, 2> binCounts{};
        for (auto it = first; it != last; ++it) {
            const sf::Vector2i bin((centroid(*it) - centroidBound.minPoint) * invBinWidth);
            for (int dim : {0, 1}) {
                binBounds[dim][axis(bin, dim)] |= prims[*it].bounds();
                ++binCounts[dim][axis(bin, dim)];
            }
        }
        // prefix and suffix sums
        std::array<std::array<AABB<float>, BINS + 1>, 2> binBoundsPrefix{};
        std::array<std::array<int, BINS + 1>, 2> binCountsPrefix{};
        std::array<std::array<AABB<float>, BINS + 1>, 2> binBoundsSuffix{};
        std::array<std::array<int, BINS + 1>, 2> binCountsSuffix{};
        for (int dim : {0, 1}) {
            for (int i = 0; i < BINS; ++i) {
                binBoundsPrefix[dim][i + 1] = binBoundsPrefix[dim][i] | binBounds[dim][i];
                binCountsPrefix[dim][i + 1] = binCountsPrefix[dim][i] + binCounts[dim][i];
            }
            for (int i = BINS - 1; i >= 0; --i) {
                binBoundsSuffix[dim][i] = binBoundsSuffix[dim][i + 1] | binBounds[dim][i];
                binCountsSuffix[dim][i] = binCountsSuffix[dim][i + 1] + binCounts[dim][i];
            }
        }
        const AABB<float> totalBound = binBoundsPrefix[0].back();
        const float totalPerimLength = totalBound.perimeterLength();

        // Compute best split. Split BINS - 1 (everything on the left) is skipped since it is
        // always worse than just making a leaf node.
        float minCost = std::numeric_limits<float>::infinity();
        int bestDim = -1;
        int bestSplit = -1;
        for (int split = 0; split < BINS - 1; ++split) {
            for (int dim : {0, 1}) {
                const float cost =
                    T_TRAV +
                    T_ISECT *
                        ((binBoundsPrefix[dim][split + 1].perimeterLength() / totalPerimLength) *
                             binCountsPrefix[dim][split + 1] +
                         (binBoundsSuffix[dim][split + 1].perimeterLength() / totalPerimLength) *
                             binCountsSuffix[dim][split + 1]);
                if (cost < minCost) {
                    minCost = cost;
                    bestDim = dim;
                    bestSplit = split;
                }
            }
        }

        const auto makeLeaf = [&] {
            nodes[bvhOffset] = BVHNode{totalBound, startIdx, numIndices, /*isLeaf=*/true};
        };
        // see if we should be leaf node
        if (numIndices * T_ISECT < minCost) {
            makeLeaf();
            return;
        }
        // repartition indices based on chosen split
        const float pivotPos =
            axis(centroidBound.minPoint, bestDim) + (bestSplit + 1) * axis(binWidth, bestDim);
        const auto mid = std::partition(
            first, last, [&](int idx) { return axis(centroid(idx), bestDim) < pivotPos; });
        const int smallerCount = static_cast<int>(mid - first);
        if (smallerCount == numIndices) {
            makeLeaf();
            return;
        }
        // internal node; children are appended after their parent
        const int childA = static_cast<int>(nodes.size());
        const int childB = childA + 1;
        nodes[bvhOffset] = BVHNode{totalBound, childA, childB, /*isLeaf=*/false};
        nodes.resize(nodes.size() + 2);  // filled in by the recursive calls
        constructNode(startIdx, smallerCount, childA);
        constructNode(startIdx + smallerCount, numIndices - smallerCount, childB);
    }

    // Refit every box bottom-up without changing the tree structure. Children are always stored
    // after their parent, so a reverse sweep visits them first.
    void fullRefit() {
        for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
            BVHNode& node = nodes[i];
            if (node.isLeaf) {
                node.bounds = {};
                for (int j = node.childAOrIdx; j < node.childAOrIdx + node.childBOrLen; ++j)
                    node.bounds |= (*primitives)[indices[j]].bounds();
            } else {
                node.bounds = nodes[node.childAOrIdx].bounds | nodes[node.childBOrLen].bounds;
            }
        }
    }

    const std::vector<Circle>* primitives;
    std::vector<int> indices;
    std::vector<BVHNode> nodes;
};

// On a hit, color is set to the color of the last circle hit.
bool intersects(sf::Vector2f point, const BVHNode& node, const BVH& bvh, Vector4f& color) {
    if (!node.bounds.intersects(point)) return false;
    if (node.isLeaf) {
        bool hit = false;
        for (int i = node.childAOrIdx; i < node.childAOrIdx + node.childBOrLen; ++i) {
            const Circle& circle = (*bvh.primitives)[bvh.indices[i]];
            if (circle.intersects(point)) {
                color = circle.color;
                hit = true;
            }
        }
        return hit;
    }
    // evaluate both children (no short-circuit) so the overlap color stays consistent
    const bool hitA = intersects(point, bvh.nodes[node.childAOrIdx], bvh, color);
    const bool hitB = intersects(point, bvh.nodes[node.childBOrLen], bvh, color);
    return hitA || hitB;
}

// SOA layout for easy buffering
struct Scene {
    std::size_t createCircle(sf::Vector2f position, float radius, const Vector4f& color,
                             sf::Vector2f direction) {
        circles.push_back({position, radius, color});
        directions.push_back(direction);  // Q3: keep directions[i] paired with circles[i]
        return circles.size() - 1;
    }

    // Q3: move every circle by its direction, bouncing off the window edges (Bonus 2)
    void update() {
        for (std::size_t i = 0; i < circles.size(); ++i) {
            Circle& c = circles[i];
            c.position += directions[i];  // move one step
            if (c.position.x - c.radius < 0 || c.position.x + c.radius > gScreenWidth) {
                directions[i].x = -directions[i].x;  // hit left/right edge: flip x direction
            }
            if (c.position.y - c.radius < 0 || c.position.y + c.radius > gScreenHeight) {
                directions[i].y = -directions[i].y;  // hit top/bottom edge: flip y direction
            }
        }
    }

    Circle& circle(std::size_t i) {
        assert(valid(i));
        return circles[i];
    }

    bool valid(std::size_t i) const { return i < circles.size(); }

    std::vector<Circle> circles;
    std::vector<sf::Vector2f> directions;  // Q3: per-circle movement per frame
};

// Bonus 1: draw a box outline in the given color
void Draw(sf::RenderWindow& window, const AABB<float>& aabb, sf::Color color) {
    // draw frame of rectangle
    static sf::RectangleShape shape;
    shape.setSize(aabb.maxPoint - aabb.minPoint);
    shape.setPosition(aabb.minPoint);
    shape.setFillColor(sf::Color::Transparent);
    shape.setOutlineColor(color);
    shape.setOutlineThickness(1.0f);
    window.draw(shape);
}

int main() {
    const sf::Vector2u screenSize(gScreenWidth, gScreenHeight);
    sf::RenderWindow window(sf::VideoMode(screenSize), "BVH", sf::State::Windowed);
    window.setFramerateLimit(30);

    std::mt19937 rng;  // default seed: the same scene every run
    const auto uniform = [&rng](float lo, float hi) {
        return std::uniform_real_distribution<float>(lo, hi)(rng);
    };
    Scene scene;
    for (int n = 0; n < 100; ++n) {
        // separate variables so the random calls happen in a fixed order
        const float radius = uniform(10, 20);
        // Bonus 2: keep the whole circle on screen by staying one radius away from the edges
        const sf::Vector2f position{uniform(radius, gScreenWidth - radius),
                                    uniform(radius, gScreenHeight - radius)};
        const Vector4f color{uniform(0, 1), uniform(0, 1), uniform(0, 1), 1};
        const sf::Vector2f direction{uniform(-2, 2), uniform(-2, 2)};  // Q3: pixels per frame
        scene.createCircle(position, radius, color, direction);
    }
    BVH bvh(scene.circles);

    // Q2: four sample offsets around each pixel for antialiasing
    constexpr float AA = 0.3333f;
    constexpr std::array<sf::Vector2f, 4> offsets = {{{-AA, -AA}, {AA, -AA}, {-AA, AA}, {AA, AA}}};

    // Bonus 1: one outline color per tree height
    const std::array<sf::Color, 7> heightColors = {
        sf::Color::Red,  sf::Color::Yellow,  sf::Color::Green, sf::Color::Cyan,
        sf::Color::Blue, sf::Color::Magenta, sf::Color::White};

    std::vector<std::uint8_t> pixels(4 * gScreenWidth * gScreenHeight);  // RGBA, starts black
    sf::Texture texture(screenSize);
    const sf::Sprite sprite(texture);

    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
            else if (const auto* key = event->getIf<sf::Event::KeyPressed>())
                if (key->code == sf::Keyboard::Key::Escape) window.close();
        }

        // Q3/Q4: move circles, then fix the BVH BEFORE tracing so boxes match the new positions
        scene.update();
        if (REBUILD) {
            bvh.reset();  // Q4: build a brand new tree
        } else {
            bvh.fullRefit();  // Q3: keep the tree, just resize the boxes
        }

        for (int y = 0; y < gScreenHeight; ++y) {
            for (int x = 0; x < gScreenWidth; ++x) {
                const sf::Vector2f pixel(static_cast<float>(x), static_cast<float>(y));
                // Q2: average four samples around the pixel to smooth edges
                Vector4f sum{};
                for (const sf::Vector2f& off : offsets) {
                    Vector4f sample{};  // reset each sample: a miss counts as transparent black
                    intersects(pixel + off, bvh.nodes[0], bvh, sample);
                    sum += sample;  // add this sample's color
                }
                const Vector4f color = sum / 4.f;  // average of the four samples
                const std::size_t i = 4 * (static_cast<std::size_t>(y) * gScreenWidth + x);
                pixels[i + 0] = static_cast<std::uint8_t>(255 * color.x);  // R
                pixels[i + 1] = static_cast<std::uint8_t>(255 * color.y);  // G
                pixels[i + 2] = static_cast<std::uint8_t>(255 * color.z);  // B
                pixels[i + 3] = static_cast<std::uint8_t>(255 * color.w);  // A
            }
        }

        texture.update(pixels.data());

        window.clear();
        window.draw(sprite);
        // Q1: only draw internal nodes, colored by height (Bonus 1)
        for (const BVHNode& node : bvh.nodes) {
            if (!node.isLeaf) {
                Draw(window, node.bounds, heightColors[node.height % heightColors.size()]);
            }
        }
        window.display();
    }
    return 0;
}
