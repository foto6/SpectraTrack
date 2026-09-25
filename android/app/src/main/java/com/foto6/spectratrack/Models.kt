package com.foto6.spectratrack

data class Detection(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
    val score: Float,
    val classId: Int,
    val label: String,
) {
    val cx: Float get() = (left + right) * 0.5f
    val cy: Float get() = (top + bottom) * 0.5f
    val width: Float get() = (right - left).coerceAtLeast(0f)
    val height: Float get() = (bottom - top).coerceAtLeast(0f)
}

data class TrackedObject(
    val id: Int,
    var left: Float,
    var top: Float,
    var right: Float,
    var bottom: Float,
    var score: Float,
    var classId: Int,
    var label: String,
    var vx: Float = 0f,
    var vy: Float = 0f,
    var age: Int = 1,
    var hits: Int = 1,
    var missed: Int = 0,
    val history: ArrayDeque<Pair<Float, Float>> = ArrayDeque(),
) {
    val cx: Float get() = (left + right) * 0.5f
    val cy: Float get() = (top + bottom) * 0.5f
    val width: Float get() = (right - left).coerceAtLeast(0f)
    val height: Float get() = (bottom - top).coerceAtLeast(0f)
}
