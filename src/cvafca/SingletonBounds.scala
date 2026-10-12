package cvafca

/** Conservative rejection for identity-projector singleton CVA models.
  * Uses the same rounded BFloat16 residuals/products, then bounds Float32
  * reduction error. A possible match always receives the original tensor score;
  * this bound never chooses a node or changes the ordering of candidates.
  */
private[cvafca] object SingletonBounds:
  private def bfloat(value: Float): Float =
    val bits = java.lang.Float.floatToRawIntBits(value)
    if (bits & 0x7f800000) == 0x7f800000 then value
    else java.lang.Float.intBitsToFloat((bits + 0x7fff + ((bits >>> 16) & 1)) & 0xffff0000)

  def mayClassify(center: Array[Float], query: Array[Float], radius: Double): Boolean =
    if center.length != query.length || radius >= 1.0 || radius.isNaN then return true
    // Deliberately exceeds the usual n*u/(1-n*u) summation bound.
    val rounding = (center.length + 2).toDouble * java.lang.Math.ulp(1.0f)
    if rounding >= 0.5 then return true
    var sum = 0.0
    var index = 0
    while index < center.length do
      if !center(index).isFinite || !query(index).isFinite then return true
      val residual = bfloat(query(index) - center(index))
      sum += bfloat(residual * residual).toDouble
      index += 1
      if index % 16 == 0 || index == center.length then
        val lower = bfloat(java.lang.Math.nextDown((sum * (1.0 - rounding)).toFloat))
        val norm = math.sqrt(math.max(0.0, lower.toDouble))
        val distance = if norm.isPosInfinity then 1.0 else norm / (1.0 + norm)
        if distance > radius then return false
    true
