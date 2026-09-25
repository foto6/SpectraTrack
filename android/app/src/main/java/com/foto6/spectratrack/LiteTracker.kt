package com.foto6.spectratrack

import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

class LiteTracker(
    private val maxMissed: Int = 10,
    private val minIou: Float = 0.12f,
    private val maxCenterRatio: Float = 1.8f,
) {
    private val tracks = linkedMapOf<Int, TrackedObject>()
    private var nextId = 1

    fun update(detections: List<Detection>): List<TrackedObject> {
        val unmatchedTracks = tracks.keys.toMutableSet()
        val unmatchedDetections = detections.indices.toMutableSet()
        val candidates = mutableListOf<Triple<Float, Int, Int>>()

        tracks.forEach { (id, track) ->
            val damping = max(0.25f, 1f - track.missed * 0.08f)
            val px1 = track.left + track.vx * damping
            val py1 = track.top + track.vy * damping
            val px2 = track.right + track.vx * damping
            val py2 = track.bottom + track.vy * damping
            val pcx = (px1 + px2) * 0.5f
            val pcy = (py1 + py2) * 0.5f
            val diag = max(hypot(track.width, track.height), 24f)

            detections.forEachIndexed { index, det ->
                if (det.classId != track.classId) return@forEachIndexed
                val iou = iou(px1, py1, px2, py2, det.left, det.top, det.right, det.bottom)
                val distRatio = hypot(det.cx - pcx, det.cy - pcy) / diag
                if (iou < minIou && distRatio > maxCenterRatio) return@forEachIndexed
                val score = iou * 2.2f + max(0f, 1f - distRatio / maxCenterRatio)
                candidates += Triple(score, id, index)
            }
        }

        candidates.sortByDescending { it.first }
        candidates.forEach { (_, id, index) ->
            if (id !in unmatchedTracks || index !in unmatchedDetections) return@forEach
            unmatchedTracks.remove(id)
            unmatchedDetections.remove(index)
            val tr = tracks.getValue(id)
            val det = detections[index]
            val oldCx = tr.cx
            val oldCy = tr.cy
            val measuredVx = det.cx - oldCx
            val measuredVy = det.cy - oldCy
            tr.vx = tr.vx * 0.6f + measuredVx * 0.4f
            tr.vy = tr.vy * 0.6f + measuredVy * 0.4f
            tr.left = det.left
            tr.top = det.top
            tr.right = det.right
            tr.bottom = det.bottom
            tr.score = det.score
            tr.label = det.label
            tr.age++
            tr.hits++
            tr.missed = 0
            appendHistory(tr, det.cx to det.cy)
        }

        unmatchedTracks.forEach { id ->
            val tr = tracks.getValue(id)
            val damping = max(0.25f, 1f - tr.missed * 0.08f)
            val dx = tr.vx * damping
            val dy = tr.vy * damping
            tr.left += dx
            tr.right += dx
            tr.top += dy
            tr.bottom += dy
            tr.age++
            tr.missed++
            appendHistory(tr, tr.cx to tr.cy)
        }

        unmatchedDetections.forEach { index ->
            val det = detections[index]
            val tr = TrackedObject(
                id = nextId++, left = det.left, top = det.top, right = det.right, bottom = det.bottom,
                score = det.score, classId = det.classId, label = det.label,
            )
            appendHistory(tr, det.cx to det.cy)
            tracks[tr.id] = tr
        }

        tracks.entries.removeIf { it.value.missed > maxMissed }
        return tracks.values.map { it.copy(history = ArrayDeque(it.history)) }
    }

    private fun appendHistory(track: TrackedObject, p: Pair<Float, Float>) {
        track.history.addLast(p)
        while (track.history.size > 36) track.history.removeFirst()
    }

    private fun iou(ax1: Float, ay1: Float, ax2: Float, ay2: Float, bx1: Float, by1: Float, bx2: Float, by2: Float): Float {
        val x1 = max(ax1, bx1)
        val y1 = max(ay1, by1)
        val x2 = min(ax2, bx2)
        val y2 = min(ay2, by2)
        val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
        val aa = max(0f, ax2 - ax1) * max(0f, ay2 - ay1)
        val ba = max(0f, bx2 - bx1) * max(0f, by2 - by1)
        return inter / max(aa + ba - inter, 1e-6f)
    }
}
